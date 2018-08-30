# -*- coding: utf-8 -*-
#!/usr/local/bin/python
import os
import sys
from copy import deepcopy
import json
import time
import cPickle
import numpy
import tensorflow as tf
import random
import math
import string
import ConfigParser
import types
import codecs 


from random import shuffle
from numpy.linalg import norm


from models import model_definition

global_var_asr_count = 1


lp = {}
lp["english"] = u"en"
lp["german"] = u"de"
lp["italian"] = u"it"
lp["russian"] = u"ru"
lp["sh"] = u"sh"
lp["bulgarian"] = u"bg"
lp["polish"] = u"pl"
lp["spanish"] = u"es"
lp["french"] = u"fr"
lp["portuguese"] = u"pt"
lp["swedish"] = u"sv"
lp["dutch"] = u"nl"


def xavier_vector(word, D=300):
    """
    Returns a D-dimensional vector for the word. 

    We hash the word to always get the same vector for the given word. 
    """

    seed_value = hash_string(word)
    numpy.random.seed(seed_value)

    neg_value = - math.sqrt(6)/math.sqrt(D)
    pos_value = math.sqrt(6)/math.sqrt(D)

    rsample = numpy.random.uniform(low=neg_value, high=pos_value, size=(D,))
    norm = numpy.linalg.norm(rsample)
    rsample_normed = rsample/norm

    return rsample_normed


def hash_string(s):

    return abs(hash(s)) % (10 ** 8)


def compare_request_lists(list_a, list_b):

    if len(list_a) != len(list_b): 
        return False

    list_a.sort()
    list_b.sort()

    for idx in range(0, len(list_a)):
        if list_a[idx] != list_b[idx]:
            return False

    return True


def evaluate_woz(evaluated_dialogues, dialogue_ontology):
    """
    Given a list of (transcription, correct labels, predicted labels), this measures joint goal (as in Matt's paper), 
    and f-scores, as presented in Shawn's NIPS paper.
    
    Assumes request is always there in the ontology.      
    """
    print_mode=True
    informable_slots = list(set(["food", "area", "price range", "prezzo", "cibo", "essen", "preisklasse", "gegend"]) & set(dialogue_ontology.keys()))
    dialogue_count = len(evaluated_dialogues)
    if "request" in dialogue_ontology:
        req_slots = [str("req_" + x) for x in dialogue_ontology["request"]]
        requestables = ["request"]
    else:
        req_slots = []
        requestables = []
    # print req_slots

    true_positives = {}
    false_negatives = {}
    false_positives = {}

    req_match = 0.0
    req_full_turn_count = 0.0

    req_acc_total = 0.0 # number of turns which express requestables
    req_acc_correct = 0.0

    for slot in dialogue_ontology:
        true_positives[slot] = 0
        false_positives[slot] = 0
        false_negatives[slot] = 0

    for value in requestables + req_slots + ["request"]:
        true_positives[value] = 0
        false_positives[value] = 0
        false_negatives[value] = 0


    correct_turns = 0 # when there is at least one informable, do all of them match?
    incorrect_turns = 0 # when there is at least one informable, if any are different. 
    
    slot_correct_turns = {}
    slot_incorrect_turns = {}

    for slot in informable_slots:
        slot_correct_turns[slot] = 0.0
        slot_incorrect_turns[slot] = 0.0

    dialogue_joint_metrics = []
    dialogue_req_metrics = []

    dialogue_slot_metrics = {}

    for slot in informable_slots:
        dialogue_slot_metrics[slot] = []


    for idx in range(0, dialogue_count):

        dialogue = evaluated_dialogues[idx]["dialogue"]
        # print dialogue

        curr_dialogue_goal_joint_total   = 0.0 # how many turns have informables
        curr_dialogue_goal_joint_correct = 0.0

        curr_dialogue_goal_slot_total = {} # how many turns in current dialogue have specific informables
        curr_dialogue_goal_slot_correct = {} # and how many of these are correct

        for slot in informable_slots:
            curr_dialogue_goal_slot_total[slot] = 0.0
            curr_dialogue_goal_slot_correct[slot] = 0.0

        creq_tp = 0.0
        creq_fn = 0.0
        creq_fp = 0.0 
        # to compute per-dialogue f-score for requestables

        for turn in dialogue:

            # first update full requestable

            req_full_turn_count += 1.0

            if requestables:

                if compare_request_lists(turn[1]["True State"]["request"], turn[2]["Prediction"]["request"]):
                    req_match += 1.0

                if len(turn[1]["True State"]["request"]) > 0:
                    req_acc_total += 1.0

                    if compare_request_lists(turn[1]["True State"]["request"], turn[2]["Prediction"]["request"]):
                        req_acc_correct += 1.0

            # per dialogue requestable metrics
            if requestables:

                true_requestables = turn[1]["True State"]["request"]
                predicted_requestables = turn[2]["Prediction"]["request"]

                for each_true_req in true_requestables:
                    if each_true_req in dialogue_ontology["request"] and each_true_req in predicted_requestables:
                        true_positives["request"] += 1
                        creq_tp += 1.0
                        true_positives["req_" + each_true_req] += 1
                    elif each_true_req in dialogue_ontology["request"]:
                        false_negatives["request"] += 1
                        false_negatives["req_" + each_true_req] += 1
                        creq_fn += 1.0
                        # print "FN:", turn[0], "---", true_requestables, "----", predicted_requestables

                for each_predicted_req in predicted_requestables:
                    # ignore matches, already counted, now need just negatives:
                    if each_predicted_req not in true_requestables:
                        false_positives["request"] += 1
                        false_positives["req_" + each_predicted_req] += 1
                        creq_fp += 1.0
                        # print "-- FP:", turn[0], "---", true_requestables, "----", predicted_requestables


            # print turn
            inf_present = {}
            inf_correct = {}

            for slot in informable_slots:
                inf_present[slot] = False
                inf_correct[slot] = True

            informable_present = False
            informable_correct = True

            for slot in informable_slots:

                try:
                    true_value = turn[1]["True State"][slot]
                    predicted_value = turn[2]["Prediction"][slot]
                except:

                    print "PROBLEM WITH", turn, "slot:", slot, "inf slots", informable_slots

                if true_value != "none":
                    informable_present = True
                    inf_present[slot] = True

                if true_value == predicted_value: # either match or none, so not incorrect
                    if true_value != "none":
                        true_positives[slot] += 1
                else:
                    if true_value == "none":
                        false_positives[slot] += 1
                    elif predicted_value == "none":
                        false_negatives[slot] += 1
                    else:
                        # spoke to Shawn - he does this as false negatives for now - need to think about how we evaluate it properly.
                        false_negatives[slot] += 1

                    informable_correct = False
                    inf_correct[slot] = False


            if informable_present:

                curr_dialogue_goal_joint_total += 1.0

                if informable_correct:
                    correct_turns += 1
                    curr_dialogue_goal_joint_correct += 1.0
                else:
                    incorrect_turns += 1

            for slot in informable_slots:
                if inf_present[slot]:
                    curr_dialogue_goal_slot_total[slot] += 1.0

                    if inf_correct[slot]:
                        slot_correct_turns[slot] += 1.0
                        curr_dialogue_goal_slot_correct[slot] += 1.0
                    else:
                        slot_incorrect_turns[slot] += 1.0


        # current dialogue requestables

        if creq_tp + creq_fp > 0.0:
            creq_precision = creq_tp / (creq_tp + creq_fp)
        else:
            creq_precision = 0.0

        if creq_tp + creq_fn > 0.0:
            creq_recall = creq_tp / (creq_tp + creq_fn)
        else:
            creq_recall = 0.0

        if creq_precision + creq_recall == 0:
            if creq_tp == 0 and creq_fn == 0 and creq_fn == 0:
                # no requestables expressed, special value
                creq_fscore = -1.0
            else:
                creq_fscore = 0.0 # none correct but some exist
        else:
            creq_fscore = (2 * creq_precision * creq_recall) / (creq_precision + creq_recall)

        dialogue_req_metrics.append(creq_fscore)


        # and current dialogue informables:

        for slot in informable_slots:
            if curr_dialogue_goal_slot_total[slot] > 0:
                dialogue_slot_metrics[slot].append(float(curr_dialogue_goal_slot_correct[slot]) / curr_dialogue_goal_slot_total[slot])
            else:
                dialogue_slot_metrics[slot].append(-1.0)

        if informable_slots:
            if curr_dialogue_goal_joint_total > 0:
                current_dialogue_joint_metric = float(curr_dialogue_goal_joint_correct) / curr_dialogue_goal_joint_total
                dialogue_joint_metrics.append(current_dialogue_joint_metric)
            else:
                # should not ever happen when all slots are used, but for validation we might not have i.e. area mentioned
                dialogue_joint_metrics.append(-1.0)

    if informable_slots:
        goal_joint_total = float(correct_turns) / float(correct_turns + incorrect_turns)
    
    slot_gj = {}
    
    total_true_positives = 0
    total_false_negatives = 0
    total_false_positives = 0

    precision = {}
    recall = {}
    fscore = {}


    # FSCORE for each requestable slot:
    if requestables:
        add_req = ["request"] + req_slots
    else:
        add_req = []

    for slot in informable_slots + add_req:
    
        if slot not in ["request"] and slot not in req_slots:
            total_true_positives += true_positives[slot]
            total_false_positives += false_positives[slot]
            total_false_negatives += false_negatives[slot]
    
        precision_denominator = (true_positives[slot] + false_positives[slot])
        
        if precision_denominator != 0:
            precision[slot] = float(true_positives[slot]) / precision_denominator
        else:
            precision[slot] = 0
        
        recall_denominator = (true_positives[slot] + false_negatives[slot])
        
        if recall_denominator != 0:
            recall[slot] = float(true_positives[slot]) / recall_denominator
        else:
            recall[slot] = 0

        if precision[slot] + recall[slot] != 0:
            fscore[slot] = (2 * precision[slot] * recall[slot]) / (precision[slot] + recall[slot])
            print "REQ - slot", slot, round(precision[slot], 3), round(recall[slot], 3), round(fscore[slot], 3)
        else:
            fscore[slot] = 0

        total_count_curr = true_positives[slot] + false_negatives[slot] + false_positives[slot]
        
        #if "req" in slot:
        #if slot in ["area", "food", "price range", "request"]:
        #print "Slot:", slot, "Count:", total_count_curr, true_positives[slot], false_positives[slot], false_negatives[slot], "[Precision, Recall, Fscore]=", round(precision[slot], 2), round(recall[slot], 2), round(fscore[slot], 2)
        #print "Slot:", slot, "TP:", true_positives[slot], "FN:", false_negatives[slot], "FP:", false_positives[slot]

    
    if requestables:

        requested_accuracy_all = req_match / req_full_turn_count

        if req_acc_total != 0:
            requested_accuracy_exist = req_acc_correct / req_acc_total
        else:
            requested_accuracy_exist = 1.0 

        slot_gj["request"] = round(requested_accuracy_exist, 3)
        #slot_gj["requestf"] = round(fscore["request"], 3)


    for slot in informable_slots:
        slot_gj[slot] = round(float(slot_correct_turns[slot]) / float(slot_correct_turns[slot] + slot_incorrect_turns[slot]), 3)

    # NIKOLA TODO: will be useful for goal joint
    if len(informable_slots) == 3:
        # print "\n\nGoal Joint: " + str(round(goal_joint_total, 3)) + "\n"
        slot_gj["joint"] = round(goal_joint_total, 3)

    if "request" in slot_gj:
        del slot_gj["request"]

    return slot_gj


def process_turn_hyp(transcription, language):
    """
    Returns the clean (i.e. handling interpunction signs) string for the given language. 
    """
    exclude = set(string.punctuation)
    exclude.remove("'")

    transcription = ''.join(ch for ch in transcription if ch not in exclude)

    transcription = transcription.lower()
    transcription = transcription.replace(u"’", "'")
    transcription = transcription.replace(u"‘", "'")
    transcription = transcription.replace("don't", "dont")
    if language == "it" or language == "italian":# or language == "en" or language == "english":
        transcription = transcription.replace("'", " ")
    if language == "en" or language == "english":# or language == "en" or language == "english":
        transcription = transcription.replace("'", "")

    return transcription


def process_woz_dialogue(woz_dialogue,language,override_en_ontology):
    """
    Returns a list of (tuple, belief_state) for each turn in the dialogue.
    """
    # initial belief state
    # belief state to be given at each turn
    if language == "english" or language == "en" or override_en_ontology:
        null_bs = {}
        null_bs["food"] = "none"
        null_bs["price range"] = "none"
        null_bs["area"] = "none"
        null_bs["request"] = []
        informable_slots = ["food", "price range", "area"]
        pure_requestables = ["address", "phone", "postcode"]

    elif (language == "italian" or language == "it"):
        null_bs = {}
        null_bs["area"] = "none"
        null_bs["cibo"] = "none"
        null_bs["prezzo"] = "none"
        null_bs["request"] = []
        informable_slots = ["cibo", "prezzo", "area"]
        pure_requestables = ["codice postale", "telefono", "indirizzo"]

    elif (language == "german" or language == "de"):
        null_bs = {}
        null_bs["gegend"] = "none"
        null_bs["essen"] = "none"
        null_bs["preisklasse"] = "none"
        null_bs["request"] = []
        informable_slots = ["essen", "preisklasse", "gegend"]
        pure_requestables = ["postleitzahl", "telefon", "adresse"]

    prev_belief_state = deepcopy(null_bs)
    dialogue_representation = []

    # user talks first, so there is no requested DA initially. 
    current_req = [""]

    current_conf_slot = [""]
    current_conf_value = [""]

    lp = {}
    lp["german"] = u"de_"
    lp["italian"] = u"it_"

    for idx, turn in enumerate(woz_dialogue):

        current_DA = turn["system_acts"]

        current_req = []
        current_conf_slot = []
        current_conf_value = []

        for each_da in current_DA:
            if each_da in informable_slots:
                current_req.append(each_da)
            elif each_da in pure_requestables:
                current_conf_slot.append("request")
                current_conf_value.append(each_da)
            else:
                if type(each_da) is list:
                    current_conf_slot.append(each_da[0])
                    current_conf_value.append(each_da[1])

        if not current_req:
            current_req = [""]

        if not current_conf_slot:
            current_conf_slot = [""]
            current_conf_value = [""]

        current_transcription = turn["transcript"]
        current_transcription = process_turn_hyp(current_transcription, language)
        
        read_asr = turn["asr"]

        current_asr = []

        for (hyp, score) in read_asr:
            current_hyp = process_turn_hyp(hyp, language)
            current_asr.append((current_hyp, score))


        old_trans = current_transcription

        exclude = set(string.punctuation)
        exclude.remove("'")

        current_transcription = ''.join(ch for ch in current_transcription if ch not in exclude)
        current_transcription = current_transcription.lower()

        current_labels = turn["turn_label"]

        current_bs = deepcopy(prev_belief_state)

        #print "=====", prev_belief_state
        if "request" in prev_belief_state:
            del prev_belief_state["request"]

        current_bs["request"] = [] # reset requestables at each turn

        for label in current_labels:
            (c_slot, c_value) = label

            if c_slot in informable_slots:
                current_bs[c_slot] = c_value

            elif c_slot == "request":
                current_bs["request"].append(c_value)

        curr_lab_dict = {}
        for x in current_labels:
            if x[0] != "request":
                curr_lab_dict[x[0]] = x[1]

        dialogue_representation.append(((current_transcription, current_asr), current_req, current_conf_slot, current_conf_value, deepcopy(current_bs), deepcopy(prev_belief_state)))

        #print "====", current_transcription, "current bs", current_bs, "past bs", prev_belief_state, "this turn update", curr_lab_dict

        prev_belief_state = deepcopy(current_bs)

    return dialogue_representation


def load_woz_data(file_path, language, percentage=1.0,override_en_ontology=False):
    """
    This method loads WOZ dataset as a collection of utterances. 

    Testing means load everything, no split. 
    """

    woz_json = json.load(codecs.open(file_path, "r", "utf-8"))
    dialogues = []

    training_turns = []

    dialogue_count = len(woz_json)

    percentage = float(percentage)
    dialogue_count = int(percentage * float(dialogue_count))

    if dialogue_count != 200:
        print "Percentage is:", percentage, "so loading:", dialogue_count

    for idx in range(0, dialogue_count):
        
        current_dialogue = process_woz_dialogue(woz_json[idx]["dialogue"], language, override_en_ontology)
        dialogues.append(current_dialogue)

        for turn_idx, turn in enumerate(current_dialogue):

            if turn_idx == 0:
                prev_turn_label = []
            else:
                prev_turn_label = current_label

            current_label = []
            
            for req_slot in turn[4]["request"]:
                current_label.append(("request", req_slot))
                #print "adding reqslot:", req_slot

            # this now includes requests:
            for inf_slot in turn[4]:
                # print (inf_slot, turn[5][inf_slot])
                if inf_slot != "request":
                    current_label.append((inf_slot, turn[4][inf_slot]))
#                    if inf_slot == "request":
#                        print "!!!!!", inf_slot, turn[5][inf_slot]

            transcription_and_asr = turn[0]
            current_utterance = (transcription_and_asr, turn[1], turn[2], turn[3], current_label, turn[5]) #turn [5] is the past belief state

            #print "$$$$", current_utterance

            training_turns.append(current_utterance)

    # print "Number of utterances in", file_path, "is:", len(training_turns)

    return (dialogues, training_turns)


def track_woz_data(dialogues, model_variables, word_vectors, dialogue_ontology, sessions):
    """
    This method evaluates the WOZ dialogues. 
    """
    evaluated_dialogues = []
    list_of_belief_states = []

    dialogue_count = len(dialogues)
    #print "DIALOGUE COUNT: ", dialogue_count

    for idx in range(0, dialogue_count):
        
        if idx % 100 == 0 and (dialogue_count == 400): # progress for test 
            print idx, "/", dialogue_count, "done."

        evaluated_dialogue, belief_states = track_dialogue_woz(model_variables, word_vectors, dialogue_ontology, dialogues[idx], sessions)
        evaluated_dialogues.append(evaluated_dialogue)
        list_of_belief_states.append(belief_states)

    dialogue_count = len(evaluated_dialogues)
    indexed_dialogues = []
    for d_idx in range(0, dialogue_count):
        new_dialogue = {}
        new_dialogue["dialogue_idx"] = d_idx
        new_dialogue["dialogue"] = evaluated_dialogues[d_idx]
        indexed_dialogues.append(new_dialogue)
   
    return indexed_dialogues, list_of_belief_states


def track_dialogue_woz(model_variables, word_vectors, dialogue_ontology, woz_dialogue, sessions):
    """
    This produces a list of belief states predicted for the given WOZ dialogue. 
    """

    prev_belief_states = {}
    belief_states = {} # for each slot, a list of numpy arrays. 

    turn_count = len(woz_dialogue)
    #print "Turn count:", turn_count

    slots_to_track = list(set(dialogue_ontology.keys()) & set(sessions.keys()) )

    for slot in slots_to_track: 
        belief_states[slot] = {}
        if slot != "request":
            value_count = len(dialogue_ontology[slot]) + 1
            prev_belief_states[slot] = numpy.zeros((value_count,), dtype="float32")

    predictions_for_dialogue = []

    # to be able to combine predictions, we must also return the belief states for each turn. So for each turn, a dictionary indexed by slot values which points to the distribution.
    list_of_belief_states = []

    #print woz_dialogue

    for idx, trans_and_req_and_label_and_currlabel in enumerate(woz_dialogue):

        list_of_belief_states.append({})

        current_bs = {}

        for slot in slots_to_track:

            if type(model_variables) is dict:
                mx = model_variables[slot]
            else:
                mx = model_variables

            #print trans_and_req_and_label_and_currlabel

            (transcription_and_asr, req_slot, conf_slot, conf_value, label, prev_belief_state) = trans_and_req_and_label_and_currlabel

            if idx == 0 or slot == "request":
                # this should put empty belief state
                example = [(transcription_and_asr, req_slot, conf_slot, conf_value, prev_belief_state)] 
            else:
                # and this has the previous prediction, the one we just made in the previous iteration. We do not want to use the right one, the one used for training. 
                example = [(transcription_and_asr, req_slot, conf_slot, conf_value, prev_bs)] 

            #print example

            transcription = transcription_and_asr[0]
            asr = transcription_and_asr[1]

            if slot == "request":

                updated_belief_state = sliding_window_over_utterance(sessions[slot], example, word_vectors, dialogue_ontology, mx, slot, print_mode=False)
                # updated_belief_state[updated_belief_state < 0.5] = 0
                list_of_belief_states[idx]["request"] = updated_belief_state

            else:
                updated_belief_state = sliding_window_over_utterance(sessions[slot], example, word_vectors, dialogue_ontology, mx, slot, print_mode=False)
                #updated_belief_state = softmax(updated_belief_state)
                #updated_belief_state = update_belief_state(prev_belief_states[slot], new_belief_state)
                prev_belief_states[slot] = updated_belief_state
                list_of_belief_states[idx][slot] = updated_belief_state

            for idx_value, value in enumerate(dialogue_ontology[slot]):
                if slot in "request":
                    current_bs[slot] = print_belief_state_woz_requestables(dialogue_ontology[slot], updated_belief_state, threshold=0.5)
                else:
                    current_bs[slot] = print_belief_state_woz_informable(dialogue_ontology[slot], updated_belief_state, threshold=0.01) # swap to 0.001 Nikola
                    #   print idx, slot, current_bs[slot], current_bs

        prev_bs = deepcopy(current_bs)

        trans_plus_sys = "User: " + transcription 
        # + req_slot, conf_slot, conf_value
        if req_slot[0] != "":
            trans_plus_sys += "    System Request: " + str(req_slot)

        if conf_slot[0] != "": 
            trans_plus_sys += "    System Confirm: " + str(conf_slot) + " " + str(conf_value)

        trans_plus_sys += "   ASR: " + str(asr)

        predictions_for_dialogue.append((trans_plus_sys, {"True State": label}, {"Prediction": current_bs}))


    return predictions_for_dialogue, list_of_belief_states


def print_belief_state_woz_informable(curr_values, distribution,threshold):
    """
    Returns the top one if it is above threshold.
    """
    max_value = "none"
    max_score = 0.0
    total_value = 0.0

    for idx, value in enumerate(curr_values):
        
        total_value += distribution[idx]
        
        if distribution[idx] >= threshold:

            if distribution[idx] >= max_score:
                max_value = value
                max_score = distribution[idx]

    if max_score >= (1.0 - total_value):
        return max_value
    else:
        return "none"


def print_belief_state_woz_requestables(curr_values, distribution, threshold):
    """
    Returns the top one if it is above threshold.
    """
    requested_slots = []

    # now we just print to JSON file:
    for idx, value in enumerate(curr_values):
        
        if distribution[idx] >= threshold:
            requested_slots.append(value)

    return requested_slots


def generate_data(utterances, word_vectors, dialogue_ontology, target_slot):
    """
    Generates a data representation we can subsequently use. 

    Let's say negative requests are now - those utterances which express no requestables. 
    """

    # each list element is a tuple with features for that utterance (unigram, bigram, trigram).
    feature_vectors = extract_feature_vectors(utterances, word_vectors)

    # indexed by slot, these two dictionaries contain lists of positive and negative examples
    # for training each slot. Each list element is (utterance_id, slot_id, value_id)
    positive_examples = {}
    negative_examples = {}

    list_of_slots = [target_slot]
    list_of_slots = dialogue_ontology.keys() #["food", "area", "price range", "request"]

    for slot_idx, slot in enumerate(list_of_slots):
        
        positive_examples[slot] = []
        negative_examples[slot] = []

        for utterance_idx, utterance in enumerate(utterances):

            slot_expressed_in_utterance = False

            # utterance[4] is the current label
            # utterance[5] is the previous one

            for (slotA, valueA) in utterance[4]:
                if slotA == slot and (valueA != "none" and valueA != []):
                    slot_expressed_in_utterance = True # if this is True, no negative examples for softmax.
                    
                    #if slot == "request":                
                    #    print slotA, valueA, utterance, utterance[4]

            if slot != "request":

                for value_idx, value in enumerate(dialogue_ontology[slot]):
                
                    if (slot, value) in utterance[4]: # utterances are ((trans, asr), sys_req_act, sys_conf, labels)
                        positive_examples[slot].append((utterance_idx, utterance, value_idx))
                        #print "POS:", utterance_idx, utterance, value_idx, value
                    else:
                        if not slot_expressed_in_utterance:
                            negative_examples[slot].append((utterance_idx, utterance, value_idx))
                            #print "NEG:", utterance_idx, utterance, value_idx, value

            elif slot == "request":

                if not slot_expressed_in_utterance:
                    negative_examples[slot].append((utterance_idx, utterance, []))
                    # print utterance[0][0], utterance[4]
                else:

                    values_expressed = []

                    for value_idx, value in enumerate(dialogue_ontology[slot]):
                        if (slot, value) in utterance[4]: # utterances are ((trans, asr), sys_req_act, sys_conf, labels)
                            values_expressed.append(value_idx)

                    positive_examples[slot].append((utterance_idx, utterance, values_expressed))
                    # print utterances[utterance_idx], "---", values_expressed


        #positive_examples[slot] = set(positive_examples[slot])
        #negative_examples[slot] = set(negative_examples[slot])
    
    return feature_vectors, positive_examples, negative_examples


def binary_mask(example, requestable_count):
    """
    takes a list, i.e. 2,3,4, and if req count is 8, returns: 00111000
    """

    zeros = numpy.zeros((requestable_count,), dtype=numpy.float32)
    for x in example:
        zeros[x] = 1
    
    return zeros


def delexicalise_utterance_values(utterance, target_slot, target_values):
    """
    Takes a list of words which represent the current utterance, the loaded vectors, finds all occurrences of both slot name and slot value, 
    and then returns the updated vector with "delexicalised tag" in them. 
    """

    if type(utterance) is list:
        utterance = " ".join(utterance)

    if target_slot == "request":
        value_count = len(target_values)
    else:
        value_count = len(target_values)+1

    delexicalised_vector = numpy.zeros((value_count,), dtype="float32")

    for idx, target_value in enumerate(target_values):

        if " " + target_value + " " in utterance:
            delexicalised_vector[idx] = 1.0

    return delexicalised_vector


def generate_examples(target_slot, feature_vectors, word_vectors, dialogue_ontology,
                      positive_examples, negative_examples, positive_count=None, negative_count=None):
    """
    This method returns a minibatch of positive_count examples followed by negative_count examples.
    If these two are not set, it creates the full dataset (used for validation and test).
    It returns: (features_unigram, features_bigram, features_trigram, features_slot, 
                 features_values, y_labels) - all we need to pass to train. 
    """

    # total number of positive and negative examples. 
    pos_example_count = len(positive_examples[target_slot])
    neg_example_count = len(negative_examples[target_slot])

    if target_slot != "request":
        label_count = len(dialogue_ontology[target_slot]) + 1 # NONE
    else:
        label_count = len(dialogue_ontology[target_slot]) 
   
    doing_validation = False
    # for validation?
    if positive_count is None:
        positive_count = pos_example_count
        doing_validation = True
    if negative_count is None: 
        negative_count = neg_example_count
        doing_validation = True

    if pos_example_count == 0 or positive_count == 0 or negative_count == 0 or neg_example_count == 0:
        print "#### SKIPPING (NO DATA): ", target_slot, pos_example_count, positive_count, neg_example_count, negative_count
        return None

    positive_indices = []
    negative_indices = []   

    if positive_count > 0:
        positive_indices = numpy.random.choice(pos_example_count, positive_count)
    else:
        print target_slot, positive_count, negative_count

    if negative_count > 0:
        negative_indices = numpy.random.choice(neg_example_count, negative_count)

    examples = []
    labels = []
    prev_labels = []

    for idx in positive_indices:
        examples.append(positive_examples[target_slot][idx])
    if negative_count > 0:
        for idx in negative_indices:
            examples.append(negative_examples[target_slot][idx])

    value_count = len(dialogue_ontology[target_slot])

    # each element of this array is (xs_unigram, xs_bigram, xs_trigram, fv_slot, fv_value):
    features_requested_slots = []
    features_confirm_slots = []
    features_confirm_values = []
    features_slot = []
    features_values = []
    features_full = []
    features_delex = []
    features_previous_state = []

    # feature vector of the used slot:
    slot_fv = word_vectors[unicode(target_slot)]

    # now go through all examples (positive followed by negative).
    for idx_example, example in enumerate(examples):

        (utterance_idx, utterance, value_idx) = example
        utterance_fv = feature_vectors[utterance_idx]
        
        # prev belief state is in utterance[5]
        prev_belief_state = utterance[5]

        if idx_example < positive_count:
            if target_slot != "request":
                labels.append(value_idx) # includes dontcare
            else:
                labels.append(binary_mask(value_idx, len(dialogue_ontology["request"])))
        else:
            if target_slot != "request":
                labels.append(value_count) # NONE - for this we need to make sure to not include utterances which express this slot
            else:
                labels.append([]) #wont ever use this

        # handling of previous labels:
        if target_slot != "request":
            prev_labels.append(prev_belief_state[target_slot])

        # for now, we just deal with the utterance, and not with WOZ data. 
        # TODO: need to get a series of delexicalised vectors, one for each value. 
        
        delex_features = delexicalise_utterance_values(utterance[0][0], target_slot, dialogue_ontology[target_slot])

        features_full.append(utterance_fv[0])
        features_requested_slots.append(utterance_fv[1])
        features_confirm_slots.append(utterance_fv[2])
        features_confirm_values.append(utterance_fv[3])
        features_delex.append(delex_features)

        prev_belief_state_vector = numpy.zeros((label_count,), dtype="float32")
        
        if target_slot != "request":

            prev_value = prev_belief_state[target_slot]

            if prev_value == "none" or prev_value not in dialogue_ontology[target_slot]:
                prev_belief_state_vector[label_count-1] = 1
            else:
                prev_belief_state_vector[dialogue_ontology[target_slot].index(prev_value)] = 1

        features_previous_state.append(prev_belief_state_vector)

    features_requested_slots = numpy.array(features_requested_slots)
    features_confirm_slots = numpy.array(features_confirm_slots)
    features_confirm_values = numpy.array(features_confirm_values)
    features_full  = numpy.array(features_full)
    features_delex = numpy.array(features_delex)
    features_previous_state = numpy.array(features_previous_state)

    y_labels = numpy.zeros((positive_count + negative_count, label_count), dtype="float32")

    for idx in range(0, positive_count):
        if target_slot != "request":
            y_labels[idx, labels[idx]] = 1
        else:
            y_labels[idx, :] = labels[idx]

    if target_slot != "request":
        y_labels[positive_count:, label_count-1] = 1# NONE, 0-indexing

    return (features_full, features_requested_slots, features_confirm_slots, \
            features_confirm_values, features_delex, y_labels, features_previous_state)


def evaluate_model(dataset_name, sess, model_variables, data, target_slot, utterances, dialogue_ontology, \
                        positive_examples, negative_examples, print_mode=False, epoch_id=""):


    start_time = time.time()

    keep_prob, x_full, x_delex, \
    requested_slots, system_act_confirm_slots, system_act_confirm_values, y_, y_past_state, accuracy, \
    f_score, precision, recall, num_true_positives, \
    num_positives, classified_positives, y, predictions, true_predictions, correct_prediction, \
    true_positives, train_step, update_coefficient = model_variables

    (xs_full, xs_sys_req, xs_conf_slots, xs_conf_values, xs_delex, xs_labels, xs_prev_labels) = data

    example_count = xs_full.shape[0]

    label_size = xs_labels.shape[1]

    batch_size = 16
    word_vector_size = 300
    longest_utterance_length = 40

    batch_count = int(math.ceil(float(example_count) / batch_size))

    total_accuracy = 0.0
    element_count = 0

    total_num_FP = 0.0 # FP
    total_num_TP = 0.0 # TP
    total_num_FN = 0.0 # FN -> prec = TP / (TP + FP), recall = TP / (TP + FN)
    total_num_TN = 0.0

    for idx in range(0, batch_count):

        left_range = idx * batch_size
        right_range = min((idx+1) * batch_size, example_count)
        curr_len = right_range - left_range # in the last batch, could be smaller than batch size

        if idx in [batch_count - 1, 0]:

            xss_full = numpy.zeros((batch_size, longest_utterance_length, word_vector_size), dtype="float32")
            xss_sys_req = numpy.zeros((batch_size, word_vector_size), dtype="float32")
            xss_conf_slots = numpy.zeros((batch_size, word_vector_size), dtype="float32")
            xss_conf_values = numpy.zeros((batch_size, word_vector_size), dtype="float32")
            xss_delex = numpy.zeros((batch_size, label_size), dtype="float32")
            xss_labels = numpy.zeros((batch_size, label_size), dtype="float32")
            xss_prev_labels = numpy.zeros((batch_size, label_size), dtype="float32")

        xss_full[0:curr_len, :, :] = xs_full[left_range:right_range, :, :]
        xss_sys_req[0:curr_len, :] = xs_sys_req[left_range:right_range, :]
        xss_conf_slots[0:curr_len, :] = xs_conf_slots[left_range:right_range, :]
        xss_conf_values[0:curr_len, :] = xs_conf_values[left_range:right_range, :]
        xss_delex[0:curr_len, :] = xs_delex[left_range:right_range, :]
        xss_labels[0:curr_len, :] = xs_labels[left_range:right_range, :]
        xss_prev_labels[0:curr_len, :] = xs_prev_labels[left_range:right_range, :]

    # ==============================================================================================

        [current_predictions, current_y, current_accuracy, update_coefficient_load] = sess.run([predictions, y, accuracy, update_coefficient], 
                         feed_dict={x_full: xss_full, x_delex: xss_delex, \
                                    requested_slots: xss_sys_req, system_act_confirm_slots: xss_conf_slots, \
                                    system_act_confirm_values: xss_conf_values, y_: xss_labels, y_past_state: xss_prev_labels, keep_prob: 1.0})

#       below lines print predictions for small batches to see what is being predicted
#        if idx == 0 or idx == batch_count - 2:
#            #print current_y.shape, xss_labels.shape, xs_labels.shape
#            print "\n\n", numpy.argmax(current_y, axis=1), "\n", numpy.argmax(xss_labels, axis=1), "\n==============================\n\n"

        total_accuracy += current_accuracy
        element_count += 1
    
    eval_accuracy = round(total_accuracy / element_count, 3)

    if print_mode:
        print "Epoch", epoch_id, "[Accuracy] = ", eval_accuracy, " ----- update coeff:", update_coefficient_load # , round(end_time - start_time, 1), "seconds. ---"

    return eval_accuracy

def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the word_vectors dictionary.
    """
    for word in word_vectors:
        word_vectors[word] /= math.sqrt((word_vectors[word]**2).sum() + 1e-6)
        word_vectors[word] = word_vectors[word] * norm
    return word_vectors


def load_word_vectors(file_destination, primary_language="english"):
    """
    This method loads the word vectors from the supplied file destination. 
    It loads the dictionary of word vectors and prints its size and the vector dimensionality. 
    """
    #print "XAVIER - returning null dictionary"
    #return {}

    print "Loading pretrained word vectors from", file_destination, "- treating", primary_language, "as the primary language."
    word_dictionary = {}

    lp = {}
    lp["english"] = u"en_"
    lp["german"] = u"de_"
    lp["italian"] = u"it_"
    lp["russian"] = u"ru_"
    lp["sh"] = u"sh_"
    lp["bulgarian"] = u"bg_"
    lp["polish"] = u"pl_"
    lp["spanish"] = u"es_"
    lp["french"] = u"fr_"
    lp["portuguese"] = u"pt_"
    lp["swedish"] = u"sv_"
    lp["dutch"] = u"nl_"

    language_key = lp[primary_language]

    f = codecs.open(file_destination, 'r', 'utf-8') 

    for line in f:
        
        line = line.split(" ", 1)   
        transformed_key = line[0].lower()

        if language_key in transformed_key:     
            
            transformed_key = transformed_key.replace(language_key, "")
        
            try:
                transformed_key = unicode(transformed_key)
            except:
                print "Can't convert the key to unicode:", transformed_key
            
            word_dictionary[transformed_key] = numpy.fromstring(line[1], dtype="float32", sep=" ")

            if word_dictionary[transformed_key].shape[0] != 300:
                print transformed_key, word_dictionary[transformed_key].shape
        
    print len(word_dictionary), "vectors loaded from", file_destination     

    return normalise_word_vectors(word_dictionary)


def extract_feature_vectors(utterances, word_vectors, ngram_size=3, longest_utterance_length=40, use_asr=False, use_transcription_in_training=False):
    """
    This method returns feature vectors for all dialogue utterances. 
    It returns a tuple of lists, where each list consists of all feature vectors for ngrams of that length.

    This method doesn't care about the labels: other methods assign actual or fake labels later on. 

    This can run on any size, including a single utterance. 

    """

    word_vector_size = random.choice(word_vectors.values()).shape[0]
    utterance_count = len(utterances)

    ngram_feature_vectors = []

    # let index 6 denote full FV (for conv net):
    for j in range(0, utterance_count):
        ngram_feature_vectors.append(numpy.zeros((longest_utterance_length * word_vector_size,), dtype="float32"))


    requested_slot_vectors = []
    confirm_slots = []
    confirm_values = []
    zero_vector = numpy.zeros((word_vector_size,), dtype="float32")

    for idx, utterance in enumerate(utterances):

        full_fv = numpy.zeros((longest_utterance_length * word_vector_size, ), dtype="float32")

        #use_asr = True

        if use_asr:
            full_asr = utterances[idx][0][1] # just use ASR
        else:
            full_asr = [(utterances[idx][0][0], 1.0)] # else create (transcription, 1.0)

        #print "Full ASR is", full_asr

        requested_slots = utterances[idx][1]
        current_requested_vector = numpy.zeros((word_vector_size,), dtype="float32")

        for requested_slot in requested_slots:
            if requested_slot != "":
                current_requested_vector += word_vectors[unicode(requested_slot)]

        requested_slot_vectors.append(current_requested_vector)

        curr_confirm_slots = utterances[idx][2] 
        curr_confirm_values = utterances[idx][3]

        current_conf_slot_vector = numpy.zeros((word_vector_size,), dtype="float32")
        current_conf_value_vector = numpy.zeros((word_vector_size,), dtype="float32")

        confirmation_count = len(curr_confirm_slots)

        for sub_idx in range(0, confirmation_count):
            current_cslot = curr_confirm_slots[sub_idx]
            current_cvalue = curr_confirm_values[sub_idx]

            if current_cslot != "" and current_cvalue != "":
                if " " not in current_cslot:
                    current_conf_slot_vector += word_vectors[unicode(current_cslot)]
                else:
                    words_in_example = current_cslot.split()
                    for cword in words_in_example:
                        current_conf_slot_vector += word_vectors[unicode(cword)]

                if " " not in current_cvalue:
                    current_conf_value_vector += word_vectors[unicode(current_cvalue)]
                else:
                    words_in_example = current_cvalue.split()
                    for cword in words_in_example:
                        current_conf_value_vector += word_vectors[unicode(cword)]

        confirm_slots.append(current_conf_slot_vector)
        confirm_values.append(current_conf_value_vector)

        asr_weighted_feature_vectors = []

        asr_count = global_var_asr_count
        asr_mass = 0.0

        for idx1 in range(0, asr_count):
            asr_mass += full_asr[idx1][1] #+ full_asr[1][1] + full_asr[2][1] + full_asr[3][1] + full_asr[4][1] 

        if use_transcription_in_training:
            transcription_mass = asr_mass - full_asr[asr_count-1][1]
            extra_example = (transcription, transcription_mass) 
            full_asr[asr_count-1] = extra_example
            asr_mass = 2 * transcription_mass


#        print "======", full_asr

        for (c_example, asr_coeff) in full_asr[0:asr_count]:

            #print c_example, asr_coeff

            full_fv = numpy.zeros((longest_utterance_length * word_vector_size, ), dtype="float32")

            if c_example != "":
                # print c_example
                words_utterance = process_turn_hyp(c_example, "en")
                words_utterance = words_utterance.split()

                for word_idx, word in enumerate(words_utterance):

                    word = unicode(word)

                    if word not in word_vectors:
                            
                        word_vectors[word] = xavier_vector(word)
                        
                        #print "== Over Utterance: Generating random word vector for", word.encode('utf-8'), ":::", numpy.sum(word_vectors[word])
                        
                    try:
                        full_fv[word_idx * word_vector_size : (word_idx+1) * word_vector_size] = word_vectors[word] 
                    except:
                        print "Something off with word:", word, word in word_vectors
                    

            asr_weighted_feature_vectors.append(numpy.reshape(full_fv, (longest_utterance_length, word_vector_size)))


        if len(asr_weighted_feature_vectors) != asr_count:
            print "££££££££££££££ Length of weighted vectors is:", len(asr_weighted_feature_vectors)    

        # list of [40, 300] into [len_list * 40, 300]
        # print len(asr_weighted_feature_vectors), asr_weighted_feature_vectors[0].shape
        ngram_feature_vectors[idx] = numpy.concatenate(asr_weighted_feature_vectors, axis=0)


    list_of_features = []

    use_external_representations = False

    for idx in range(0, utterance_count):
        list_of_features.append((ngram_feature_vectors[idx],
                                 requested_slot_vectors[idx],
                                 confirm_slots[idx], 
                                 confirm_values[idx],
                                 ))

    return list_of_features


def train_run(target_language, override_en_ontology, percentage, model_type, dataset_name, word_vectors, exp_name, dialogue_ontology, model_variables, target_slot, language="en", max_epoch=20, batches_per_epoch=4096, batch_size=256):
    """
    This method trains a model on the data and saves the file parameters to a file which can 
    then be loaded to do evaluation. 
    """

    keep_prob, x_full, x_delex, \
    requested_slots, system_act_confirm_slots, system_act_confirm_values, \
    y_, y_past_state, accuracy, \
    f_score, precision, recall, num_true_positives, \
    num_positives, classified_positives, y, predictions, true_predictions,  \
    correct_prediction, true_positives, train_step, update_coefficient = model_variables

    slots = dialogue_ontology.keys()

    _, utterances_train2 = load_woz_data("data/" + dataset_name + "/" + dataset_name + "_train_" + language + ".json", language) # parameter determines which ones are loaded
    
    utterance_count = len(utterances_train2)

    _, utterances_val2 = load_woz_data("data/" + dataset_name + "/" + dataset_name + "_validate_" + language + ".json", language)

    val_count = len(utterances_val2)

    utterances_train = utterances_train2 + utterances_val2[0:int(0.75 * val_count)]
    utterances_val = utterances_val2[int(0.75 * val_count):]

    print "\nTraining using:", dataset_name, " data - Utterance count:", utterance_count


    # training feature vectors and positive and negative examples list. 
    print "Generating data for training set:"
    feature_vectors, positive_examples, negative_examples = generate_data(utterances_train, word_vectors, dialogue_ontology, target_slot)

    print "Generating data for validation set:"
    # same for validation (can pre-compute full representation, will be used after each epoch):
    fv_validation, positive_examples_validation, negative_examples_validation = \
                generate_data(utterances_val, word_vectors, dialogue_ontology, target_slot)

    val_data = generate_examples(target_slot, fv_validation, word_vectors, dialogue_ontology,
                                positive_examples_validation, negative_examples_validation)


    if val_data is None:
        print "val data is none"
        return

    # will be used to save model parameters with best validation scores.  
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print_mode = False
    # Model training:

    best_f_score = -0.01

    print "\nDoing", batches_per_epoch, "randomly drawn batches of size", batch_size, "for", max_epoch, "training epochs.\n"
    start_time = time.time()

    ratio = {}
    
    for slot in dialogue_ontology:
        if slot not in ratio:
            ratio[slot] = batch_size / 2 # fewer negatives
    
    epoch = 0
    last_update = -1

    while epoch < max_epoch:
        
        sys.stdout.flush()

        epoch += 1
        current_epoch_fscore = 0.0
        current_epoch_acc = 0.0

        if epoch > 1 and target_slot == "request":
            return

        for batch_id in range(batches_per_epoch):
            
            random_positive_count = ratio[target_slot]
            random_negative_count = batch_size - random_positive_count

            batch_data = generate_examples(target_slot, feature_vectors, word_vectors, dialogue_ontology,
                positive_examples, negative_examples, random_positive_count, random_negative_count)

            (batch_xs_full, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, 
                batch_delex, batch_ys, batch_ys_prev) = batch_data

            [_, cf, cp, cr, ca] = sess.run([train_step, f_score, precision, recall, accuracy], feed_dict={x_full: batch_xs_full, \
                                              x_delex: batch_delex, \
                                              requested_slots: batch_sys_req, \
                                              system_act_confirm_slots: batch_sys_conf_slots, \
                                              system_act_confirm_values: batch_sys_conf_values, \
                                              y_: batch_ys, y_past_state: batch_ys_prev, keep_prob: 0.5})


        # ================================ VALIDATION ==============================================

        epoch_print_step = 1
        if epoch % 5 == 0 or epoch == 1:
            if epoch == 1:
                print "Epoch", "0", "to", epoch, "took", round(time.time() - start_time, 2), "seconds."

            else:
                print "Epoch", epoch-5, "to", epoch, "took", round(time.time() - start_time, 2), "seconds."
                start_time = time.time()

        current_f_score = evaluate_model(dataset_name, sess, model_variables, val_data, target_slot, utterances_val, \
        dialogue_ontology, positive_examples_validation, negative_examples_validation, print_mode=True, epoch_id=epoch+1)

        stime = time.time()
        current_metric = current_f_score
        print " Validation metric for slot:", target_slot, " :", round(current_metric, 5), " eval took", round(time.time() - stime, 2), "last update at:", last_update, "/", max_epoch

        # and if we got a new high score for validation f-score, we need to save the parameters:
        if current_metric > best_f_score:

            last_update = epoch

            if epoch < 100:
                if int(epoch * 1.5) > max_epoch:
                    max_epoch = int(epoch * 1.5) 
                    # print "Increasing max epoch to:", max_epoch
            else:
                if int(epoch * 1.2) > max_epoch:
                    max_epoch = int(epoch * 1.2) 
                    # print "Increasing max epoch to:", max_epoch


            print "\n ====================== New best validation metric:", round(current_metric, 4),  \
                  " - saving these parameters. Epoch is:", epoch + 1, "/", max_epoch, "---------------- ===========  \n"

            best_f_score = current_metric
            path_to_save = "./models/" + model_type + "_" + language + "_" +  str(override_en_ontology) + "_" + \
                   str(dataset_name) + "_" + str(target_slot)+ "_" + str(exp_name) + "_" + str(percentage) + ".ckpt"

            save_path = saver.save(sess, path_to_save)

    print "The best parameters achieved a validation metric of", round(best_f_score, 4)
    

def print_slot_predictions(distribution, slot_values, target_slot, threshold=0.05):
    """
    Prints all the activated slot values for the provided predictions
    """

    predicted_values = []
    for idx, value in enumerate(slot_values):
        if distribution[idx] >= threshold:
            predicted_values += ((value, round(distribution[idx], 2) )) #, round(post_sf[idx], 2) ))
    
    print "Predictions for", str(target_slot + ":"), predicted_values

    return predicted_values


def return_slot_predictions(distribution, slot_values, target_slot, threshold=0.05):
    """
    Prints all the activated slot values for the provided predictions
    """
    predicted_values = []
    for idx, value in enumerate(slot_values):
        if distribution[idx] >= threshold:
            predicted_values += ((value, round(distribution[idx], 2) )) #, round(post_sf[idx], 2) ))
    
    return predicted_values


def sliding_window_over_utterance(sess, utterance, word_vectors, dialogue_ontology, model_variables, target_slot, print_mode=True):
    """
    """

    if type(model_variables) is dict:
        model_variables = model_variables[target_slot]

    list_of_outputs = test_utterance(sess, utterance, word_vectors, dialogue_ontology, model_variables, target_slot, print_mode)
    belief_state = list_of_outputs[0]

    return belief_state


def test_utterance(sess, utterances, word_vectors, dialogue_ontology, model_variables, target_slot, do_print=True):
    """
    Returns a list of belief states, to be weighted later. 
    """

    potential_values = dialogue_ontology[target_slot]

    if target_slot == "request":
        value_count = len(potential_values)
    else:
        value_count = len(potential_values) + 1

    # should be a list of features for each ngram supplied. 
    fv_tuples = extract_feature_vectors(utterances, word_vectors, use_asr=True)
    utterance_count = len(utterances)

    belief_state = numpy.zeros((value_count,), dtype="float32")

    # accumulators
    slot_values = []
    candidate_values = []
    delexicalised_features = []
    fv_full = []
    fv_sys_req = []
    fv_conf_slot = []
    fv_conf_val = []
    features_previous_state = []


    for idx_hyp, extracted_fv in enumerate(fv_tuples):

        current_utterance = utterances[idx_hyp][0][0]

        prev_belief_state = utterances[idx_hyp][4]

        prev_belief_state_vector = numpy.zeros((value_count,), dtype="float32")
        
        if target_slot != "request":

            prev_value = prev_belief_state[target_slot]

            if prev_value == "none" or prev_value not in dialogue_ontology[target_slot]:
                prev_belief_state_vector[value_count-1] = 1
            else:
                prev_belief_state_vector[dialogue_ontology[target_slot].index(prev_value)] = 1

        features_previous_state.append(prev_belief_state_vector)

        (full_utt, sys_req, conf_slot, conf_value) = extracted_fv 

        delex_vector = delexicalise_utterance_values(current_utterance, target_slot, dialogue_ontology[target_slot])

        fv_full.append(full_utt)
        delexicalised_features.append(delex_vector)
        fv_sys_req.append(sys_req)
        fv_conf_slot.append(conf_slot)
        fv_conf_val.append(conf_value)

    slot_values = numpy.array(slot_values)
    candidate_values = numpy.array(candidate_values)
    delexicalised_features = numpy.array(delexicalised_features) # will be [batch_size, label_size, longest_utterance_length, vector_dimension]

    fv_sys_req = numpy.array(fv_sys_req)
    fv_conf_slot = numpy.array(fv_conf_slot)
    fv_conf_val = numpy.array(fv_conf_val)
    fv_full = numpy.array(fv_full)
    features_previous_state = numpy.array(features_previous_state)

    keep_prob, x_full, x_delex, \
    requested_slots, system_act_confirm_slots, system_act_confirm_values, y_, y_past_state, accuracy, \
    f_score, precision, recall, num_true_positives, \
    num_positives, classified_positives, y, predictions, true_predictions, correct_prediction, \
    true_positives, train_step, update_coefficient = model_variables

    distribution, update_coefficient_load = sess.run([y, update_coefficient], feed_dict={x_full: fv_full, x_delex: delexicalised_features, \
                                      requested_slots: fv_sys_req, \
                                      system_act_confirm_slots: fv_conf_slot, y_past_state: features_previous_state, system_act_confirm_values: fv_conf_val, \
                                      keep_prob: 1.0})


    belief_state = distribution[:, 0]

    current_start_idx = 0
    list_of_belief_states = []

    for idx in range(0, utterance_count):
        current_distribution = distribution[idx, :]
        list_of_belief_states.append(current_distribution)

    if do_print:
        print_slot_predictions(list_of_belief_states[0], potential_values, target_slot, threshold=0.1)

    if len(list_of_belief_states) == 1:
        return [list_of_belief_states[0]]

    return list_of_belief_states


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    sf = numpy.exp(x)
    sf = sf/numpy.sum(sf, axis=0)
    return sf


class NeuralBeliefTracker:
    """
    Call to initialise the model with pre-trained parameters and given ontology. 
    """
    def __init__(self, config_filepath):

        config = ConfigParser.RawConfigParser()
        self.config = config

        try:
            config.read(config_filepath)
        except:
            print "Couldn't read config file from", config_filepath, "... aborting."
            return None

        dataset_name = config.get("model", "dataset_name")

        word_vectors = {}
        word_vector_destination = config.get("data", "word_vectors") 

        lp = {}
        lp["english"] = u"en"
        lp["german"] = u"de"
        lp["italian"] = u"it"
       
        try:
            language = config.get("model", "language")
            language_suffix = lp[language]
        except:
            language = "english"
            language_suffix = lp[language]

        self.language = language
        self.language_suffix = language_suffix

        self.num_models = int(config.get("model", "num_models"))

        self.batches_per_epoch = int(config.get("train", "batches_per_epoch"))
        self.max_epoch = int(config.get("train", "max_epoch"))
        self.batch_size = int(config.get("train", "batch_size"))


        if not os.path.isfile(word_vector_destination): 
            print "Vectors not there, downloading small Paragram and putting it there."
            os.system("mkdir -p word-vectors/")
            os.system("mkdir -p models/")
            os.system("mkdir -p results/")
            os.system("wget -O word-vectors/prefix_paragram.txt https://www.dropbox.com/s/r35ih722bbjpn8b/prefix_paragram.txt?dl=0")
            word_vector_destination = "word-vectors/prefix_paragram.txt"


        word_vectors = load_word_vectors(word_vector_destination, primary_language=language)

        word_vectors["tag-slot"] = xavier_vector("tag-slot")
        word_vectors["tag-value"] = xavier_vector("tag-value")

        ontology_filepath = config.get("model", "ontology_filepath") 
        dialogue_ontology = json.load(codecs.open(ontology_filepath, "r", "utf-8"))
        dialogue_ontology = dialogue_ontology["informable"]
        slots = dialogue_ontology.keys()

        word_vector_size = random.choice(word_vectors.values()).shape[0]

        # a bit of hard-coding to make our lives easier. 
        if u"price" in word_vectors and u"range" in word_vectors:
            word_vectors[u"price range"] = word_vectors[u"price"] + word_vectors[u"range"]
        if u"post" in word_vectors and u"code" in word_vectors:
            word_vectors[u"postcode"] = word_vectors[u"post"] + word_vectors[u"code"]
        if u"dont" in word_vectors and u"care" in word_vectors:
            word_vectors[u"dontcare"] = word_vectors[u"dont"] + word_vectors[u"care"]
        if u"addressess" in word_vectors:
            word_vectors[u"addressess"] = word_vectors[u"addresses"]
        if u"dont" in word_vectors:
            word_vectors[u"don't"] = word_vectors[u"dont"]
            
        if language == "italian":
            word_vectors["dontcare"] = word_vectors["non"] + word_vectors["importa"]
            word_vectors["non importa"] = word_vectors["non"] + word_vectors["importa"]
        
        if language == "german":
            word_vectors["dontcare"] = word_vectors["es"] + word_vectors["ist"] + word_vectors["egal"]
            word_vectors["es ist egal"] = word_vectors["es"] + word_vectors["ist"] + word_vectors["egal"]

        exp_name = config.get("data", "exp_name")

        config_model_type = config.get("model", "model_type")
        use_cnn  = False

        if config_model_type == "cnn":
            print "----------- Config Model Type:", config_model_type, "-------------"
            use_cnn = True
            model_type = "CNN"
        elif config_model_type == "dnn":
            print "----------- Config Model Type:", config_model_type, "-------------"
            model_type = "DNN"

        self.value_specific_decoder = config.get("model", "value_specific_decoder")

        if self.value_specific_decoder in ["True", "true"]:
            self.value_specific_decoder = True
        else:
            self.value_specific_decoder = False



        self.learn_belief_state_update = config.get("model", "learn_belief_state_update")

        if self.learn_belief_state_update in ["True", "true"]:
            self.learn_belief_state_update = True
        else:
            self.learn_belief_state_update = False

        print "value_specific_decoder", self.value_specific_decoder
        print "learn_belief_state_update", self.learn_belief_state_update

        dontcare_value = "dontcare"
        if language == "italian":
            dontcare_value = "non importa"
        if language == "german":
            dontcare_value = "es ist egal"

        for slot_name in slots:
            if dontcare_value not in dialogue_ontology[slot_name] and slot_name != "request":
                dialogue_ontology[slot_name].append(dontcare_value)
            for value in dialogue_ontology[slot_name]:
                value = unicode(value)
                if u" " not in value and value not in word_vectors:

                    word_vectors[unicode(value)] = xavier_vector(unicode(value))
                    print "-- Generating word vector for:", value.encode("utf-8"), ":::", numpy.sum(word_vectors[value])

        # add up multi-word word values to get their representation:
        for slot in dialogue_ontology.keys():
            if " " in slot:
                slot = unicode(slot)
                word_vectors[slot] = numpy.zeros((word_vector_size,), dtype="float32")
                constituent_words = slot.split()
                for word in constituent_words:
                    word = unicode(word)
                    if word in word_vectors:
                        word_vectors[slot] += word_vectors[word]

            for value in dialogue_ontology[slot]:
                if " " in value:
                    value = unicode(value)
                    word_vectors[value] = numpy.zeros((word_vector_size,), dtype="float32")
                    constituent_words = value.split()
                    for word in constituent_words:
                        word = unicode(word)
                        if word in word_vectors:
                            word_vectors[value] += word_vectors[word]

        self.use_delex_features = config.get("model", "delex_features")

        if self.use_delex_features in ["True", "true"]:
            self.use_delex_features = True
        else:
            self.use_delex_features = False

        # Neural Net Initialisation (keep variables packed so we can move them to either method):
        self.model_variables = {}

        for slot in dialogue_ontology:
            print "Initialisation of model variables for slot:", slot
            if slot == "request":
                
                slot_vectors = numpy.zeros((len(dialogue_ontology[slot]), 300), dtype="float32")
                value_vectors = numpy.zeros((len(dialogue_ontology[slot]), 300), dtype="float32")

                for value_idx, value in enumerate(dialogue_ontology[slot]):
                        slot_vectors[value_idx, :] = word_vectors[slot]
                        value_vectors[value_idx, :] = word_vectors[value]

                self.model_variables[slot] = model_definition(word_vector_size, len(dialogue_ontology[slot]), slot_vectors, value_vectors, \
                     use_delex_features=self.use_delex_features, use_softmax=False, value_specific_decoder=self.value_specific_decoder, learn_belief_state_update=self.learn_belief_state_update)
            else:
                
                slot_vectors = numpy.zeros((len(dialogue_ontology[slot])+1, 300), dtype="float32") # +1 for None
                value_vectors = numpy.zeros((len(dialogue_ontology[slot])+1, 300), dtype="float32")

                for value_idx, value in enumerate(dialogue_ontology[slot]):
                        slot_vectors[value_idx, :] = word_vectors[slot]
                        value_vectors[value_idx, :] = word_vectors[value]

                self.model_variables[slot] = model_definition(word_vector_size, len(dialogue_ontology[slot]), slot_vectors, value_vectors, use_delex_features=self.use_delex_features, \
                                                 use_softmax=True, value_specific_decoder=self.value_specific_decoder, learn_belief_state_update=self.learn_belief_state_update)
        
        self.dialogue_ontology = dialogue_ontology
        
        self.model_type = model_type
        self.dataset_name = dataset_name
        
        self.exp_name = exp_name
        self.word_vectors = word_vectors

    def track_utterance(self, current_utterance, req_slot="", conf_slot="", conf_value="", past_belief_state=None):
        """
        Returns a dictionary with predictions for values in the current ontology.
        """ 
        utterance = current_utterance.decode("utf-8")
        utterance = unicode(utterance.lower())
        utterance = utterance.replace(u".", u" ")  
        utterance = utterance.replace(u",", u" ")  
        utterance = utterance.replace(u"?", u" ")  
        utterance = utterance.replace(u"-", u" ")
        utterance = utterance.strip()  

        if past_belief_state is None:
            past_belief_state = {"food": "none", "area": "none", "price range": "none"}


        utterance = [((utterance, [(utterance, 1.0)]), [req_slot], [conf_slot], [conf_value], past_belief_state)]

        print "Testing Utterance: ", utterance

        saver = tf.train.Saver()
        sess = tf.Session()

        prediction_dict = {}
        belief_states = {}
        current_bs = {}

        for slot in self.dialogue_ontology:

            try:
                path_to_load = "models/" + self.model_type + "_en_False_" + \
                    str(self.dataset_name) + "_" + str(slot)+ "_" + str(self.exp_name) + "_1.0.ckpt"

                saver.restore(sess, path_to_load)

            except:
                print "Can't restore for slot", slot, " - from file:", path_to_load
                return

            belief_state = sliding_window_over_utterance(sess, utterance, self.word_vectors, self.dialogue_ontology, self.model_variables, slot)
            belief_states[slot] = belief_state

            # Nikola - red flag, this print could be important. 
            predicted_values = return_slot_predictions(belief_state, self.dialogue_ontology[slot], slot, 0.5)
            prediction_dict[slot] = predicted_values

            current_bs[slot] = print_belief_state_woz_informable(self.dialogue_ontology[slot], belief_state, threshold=0.0) # swap to 0.001 Nikola

        return prediction_dict, current_bs


    def train(self):
        """
        FUTURE: Train the NBT model with new dataset.
        """
        for slot in self.dialogue_ontology.keys():
            print "\n==============  Training the NBT Model for slot", slot, "===============\n"
            stime = time.time()
            train_run(target_language=self.language, override_en_ontology=False, percentage=1.0, model_type="CNN", dataset_name=self.dataset_name, \
                    word_vectors=self.word_vectors, exp_name=self.exp_name, dialogue_ontology=self.dialogue_ontology, model_variables=self.model_variables[slot], target_slot=slot, language=self.language_suffix, \
                    max_epoch=self.max_epoch, batches_per_epoch=self.batches_per_epoch, batch_size=self.batch_size)
            print "\n============== Training this model took", round(time.time()-stime, 1), "seconds. ==================="


    def test_woz(self):

        override_en_ontology = False
        percentage = 1.0

        woz_dialogues, training_turns = load_woz_data("data/" + self.dataset_name + "/" + self.dataset_name + "_test_" + self.language_suffix + ".json", self.language, override_en_ontology=False)
        
        sessions = {}
        saver = tf.train.Saver()

        print "WOZ evaluation using language:", self.language, self.language_suffix

        sessions = {}
        saver = tf.train.Saver()

        list_of_belief_states = []

        for model_id in range(0, self.num_models):

            if self.language == "english" or self.language == "en" or override_en_ontology:
                slots_to_load = ["food", "price range", "area", "request"]
            elif self.language == "italian" or self.language == "it":
                slots_to_load = ["cibo", "prezzo", "area", "request"]
            elif self.language == "german" or self.language == "de":
                slots_to_load = ["essen", "preisklasse", "gegend", "request"]


            for load_slot in slots_to_load:

                path_to_load = "./models/" + self.model_type + "_" + self.language_suffix + "_" + str(override_en_ontology) + "_" + \
                                       self.dataset_name + "_" + str(load_slot)+ "_" + str(self.exp_name) + "_" + str(percentage) + ".ckpt"

                print "----------- Loading Model", path_to_load, " ----------------"

                sessions[load_slot] = tf.Session()
                saver.restore(sessions[load_slot], path_to_load)

            evaluated_dialogues, belief_states = track_woz_data(woz_dialogues, self.model_variables, self.word_vectors, self.dialogue_ontology, sessions)
            list_of_belief_states.append(belief_states) # only useful for interpolating. 

        results = evaluate_woz(evaluated_dialogues, self.dialogue_ontology)

        json.dump(evaluated_dialogues, open("results/woz_tracking.json", "w"), indent=4)                
        
        print json.dumps(results, indent=4)


def main():

    try: 
        config_filepath = sys.argv[2]
    except:
        print "Config not specified."  
        return

    NBT = NeuralBeliefTracker(config_filepath)

    do_training = False
    do_woz = False

    try: 
        switch = sys.argv[1]
        if switch == "train":
            do_training = True
        elif switch == "woz":
            do_woz = True
    except:
        print "Training/Testing not specified, defaulting to input testing."  

    if do_training:
        NBT.train()
    elif do_woz:
        NBT.test_woz()
    else:
        previous_belief_state = None

        while True:

            req_slot = raw_input("Enter system requrement slot:")
            conf_slot = raw_input("Enter system confirm slot:")
            conf_value = raw_input("Enter system confirm value:")
            
            if req_slot not in NBT.dialogue_ontology:
                print req_slot, "---", NBT.dialogue_ontology.keys()
                req_slot = ""
            else:
                req_slot = req_slot

            if  conf_slot not in NBT.dialogue_ontology:
                conf_slot = ""
                conf_value = ""
            elif conf_value not in NBT.dialogue_ontology[conf_slot]:
                conf_slot = ""
                conf_value = ""
            else:
                conf_slot = conf_slot
                conf_value = conf_value

            utterance = raw_input("Enter utterance for prediction:")

            if "reset" in utterance:
                previous_belief_state = None
            else:
                predictions, previous_belief_state = NBT.track_utterance(utterance, req_slot=req_slot, conf_slot=conf_slot, conf_value=conf_value, past_belief_state=previous_belief_state)
                print json.dumps(predictions, indent=4)


if __name__ == "__main__":
    main()              

