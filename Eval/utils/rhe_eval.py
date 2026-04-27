from collections import defaultdict


def tsinghua_F1(pred, gold, edu_num):
    cnt_golden, cnt_pred, cnt_cor_bi, cnt_cor_multi = 0, 0, 0, 0
    for hypothesis, reference, edu_num in zip(pred, gold, edu_num):
        cnt = [0] * edu_num
        for r in reference:
            cnt[r[1]] += 1
        for i in range(edu_num):
            if cnt[i] == 0:
                cnt_golden += 1
        cnt_pred += 1
        if cnt[0] == 0:
            cnt_cor_bi += 1
            cnt_cor_multi += 1
        cnt_golden += len(reference)
        cnt_pred += len(hypothesis)
        for pair in hypothesis:
            if pair in reference:
                cnt_cor_bi += 1
                if hypothesis[pair] == reference[pair]:
                    cnt_cor_multi += 1
    prec_bi, recall_bi = cnt_cor_bi * 1. / cnt_pred, cnt_cor_bi * 1. / cnt_golden
    f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
    prec_multi, recall_multi = cnt_cor_multi * 1. / cnt_pred, cnt_cor_multi * 1. / cnt_golden
    f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    print('link precision is {}'.format(prec_bi))
    return f1_bi, f1_multi


def compute_the_Accuracy_Of_Different_RelaType(predict_list_dic):
    """
    计算不同关系类型的正确率
    :return:
    """
    total_relation_dic = {}
    correct_relation_dic = {}
    predictlabelset = set()
    total_ref_rel_num = 0
    for id, item in predict_list_dic.items():
        hypo = item['hypothesis']
        ref = item['reference']
        total_ref_rel_num += len(ref)
        for _,rel in hypo.items():
            predictlabelset.add(rel)
        for ref_link, ref_rel in ref.items():
            if ref_rel in total_relation_dic:
                total_relation_dic [ref_rel] += 1
            else:
                total_relation_dic [ref_rel] = 1
            if ref_link in hypo and ref_rel == hypo[ref_link]:
                if ref_rel in correct_relation_dic:
                    correct_relation_dic[ref_rel] += 1
                else:
                    correct_relation_dic[ref_rel] = 1
    label2id = {'Comment': 0, 'Clarification_question': 1, 'Elaboration': 2, 'Acknowledgement': 3, 'Explanation': 4,
                'Conditional': 5,
                'Question-answer_pair': 6, 'Alternation': 7, 'Q-ELab': 8, 'Result': 9, 'Background': 10,
                'Narration': 11,
                'Correction': 12, 'Parallel': 13, 'Contrast': 14, 'Continuation': 15}


    id2label = {}
    for key, value in label2id.items():
        id2label[value] = key
    new_correct_relation_dic  = {}
    new_total_relation_dic  = {}
    for id,value in correct_relation_dic.items():
        new_correct_relation_dic[id] = value

    for id, value in total_relation_dic.items():
        new_total_relation_dic[id] = value
    print(new_correct_relation_dic)
    print(new_total_relation_dic)
    print(total_ref_rel_num)
    print('percentage of relation type')
    for key, value in new_total_relation_dic.items():
        print(key , round(value/total_ref_rel_num,4))

    print('accuracy of different relation type')
    for key, value in new_correct_relation_dic.items():
        print(key + ':' + str(round(value/new_total_relation_dic[key],4)))