"""
    This is a basic evaluation function that'd give evaluation of tagger and
    parser. For parser, it provides labeled and unlabeled evalution.
"""


def filter_get_P_R_F1(gts, preds, ignore_list = []):
    """
        This function will remove all to be ignored labels
        and get prec, recall, F1
    """

    ## removing labels which are to be ignored
    preds = [pred for pred in preds if pred.split('-')[-1] not in ignore_list]    
    gts = [gt for gt in gts if gt.split('-')[-1] not in ignore_list]    

    ## calculating P,R,F1
    num_overlap = len([t for t in preds if t in gts])
    precision = num_overlap / len(preds) if len(preds) > 0 else 0.0
    recall = num_overlap / len(gts) if len(gts) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return round(precision,4), round(recall, 4), round(f1, 4)

def evaluate_model(model_output, label_to_class_map, ignore_tags = [], ignore_edges = []):
    """
        model_output: This is output of get_output_as_list_of_dicts() function, which contains
        information about both, ground truth and predictions. Must be a list. 
    """

    ## get indices of ignored tags/edges
    ignore_tag_indices = [str(label_to_class_map['tag2class'][tag]) for tag in ignore_tags]
    ignore_edge_indices = [str(label_to_class_map['edgelabel2class'][edge]) for edge in ignore_edges]
    ignore_head_index = ['0']

    ## tagger's output
    tagger_preds = [ f'{i}-{j}-{word}-{pred_tag}' for i, elem in enumerate(model_output) for j, (word, pred_tag) in enumerate(zip(elem['words'], elem['pos_tags_pred']))]
    tagger_gts = [ f'{i}-{j}-{word}-{gt_tag}' for i, elem in enumerate(model_output) for j, (word, gt_tag) in enumerate(zip(elem['words'], elem['pos_tags_gt'])) ]
    tagger_results = {}
    tagger_results['P'], tagger_results['R'], tagger_results['F1'] = filter_get_P_R_F1(tagger_gts, tagger_preds, ignore_list = ignore_tag_indices)
    
    ## parser labeled output
    parser_labeled_pred = [ f'{i}-{j}-{word}-{head_pred}-{edge_pred}' for i, elem in enumerate(model_output) for j, (word, edge_pred, head_pred) in enumerate(zip(elem['words'], elem['head_tags_pred'], elem['head_indices_pred']))]
    parser_labeled_gt = [ f'{i}-{j}-{word}-{head_gt}-{edge_gt}' for i, elem in enumerate(model_output) for j, (word, edge_gt, head_gt) in enumerate(zip(elem['words'], elem['head_tags_gt'], elem['head_indices_gt'])) ]
    parser_labeled_results = {}
    parser_labeled_results['P'], parser_labeled_results['R'], parser_labeled_results['F1'] = filter_get_P_R_F1(parser_labeled_gt, parser_labeled_pred, ignore_list = ignore_edge_indices)

    ## parser unlabeled output
    parser_unlabeled_pred = [ f'{i}-{j}-{word}-{head_pred}' for i, elem in enumerate(model_output) for j, (word, head_pred) in enumerate(zip(elem['words'], elem['head_indices_pred']))]
    parser_unlabeled_gt = [ f'{i}-{j}-{word}-{head_gt}' for i, elem in enumerate(model_output) for j, (word, head_gt) in enumerate(zip(elem['words'], elem['head_indices_gt']))]
    parser_unlabeled_results = {}
    parser_unlabeled_results['P'], parser_unlabeled_results['R'], parser_unlabeled_results['F1'] = filter_get_P_R_F1(parser_unlabeled_gt, parser_unlabeled_pred, ignore_list = ignore_head_index)

    return {'tagger_results' :  tagger_results, 'parser_labeled_results': parser_labeled_results, 'parser_unlabeled_results' : parser_unlabeled_results}

def evaluate_model_with_all_labels(model_output, label_to_class_map, ignore_tags = [], ignore_edges = []):
    """
        model_output: we evaluate the model with all the labels except "no_label", if it exists. 
        So anything that comes to ignore_edges and ignore_tags variables, will be unused!
    """

    ## get indices of ignored tags/edges
    ignore_tag_indices = [str(label_to_class_map['tag2class'][tag]) for tag in ['no_label']]
    ignore_edge_indices = [str(label_to_class_map['edgelabel2class'][edge]) for edge in ['no_label']]
    ignore_head_index = []

    ## tagger's output
    tagger_preds = [ f'{i}-{j}-{word}-{pred_tag}' for i, elem in enumerate(model_output) for j, (word, pred_tag) in enumerate(zip(elem['words'], elem['pos_tags_pred']))]
    tagger_gts = [ f'{i}-{j}-{word}-{gt_tag}' for i, elem in enumerate(model_output) for j, (word, gt_tag) in enumerate(zip(elem['words'], elem['pos_tags_gt'])) ]
    tagger_results = {}
    tagger_results['P'], tagger_results['R'], tagger_results['F1'] = filter_get_P_R_F1(tagger_gts, tagger_preds, ignore_list = ignore_tag_indices)
    
    ## parser labeled output
    parser_labeled_pred = [ f'{i}-{j}-{word}-{head_pred}-{edge_pred}' for i, elem in enumerate(model_output) for j, (word, edge_pred, head_pred) in enumerate(zip(elem['words'], elem['head_tags_pred'], elem['head_indices_pred']))]
    parser_labeled_gt = [ f'{i}-{j}-{word}-{head_gt}-{edge_gt}' for i, elem in enumerate(model_output) for j, (word, edge_gt, head_gt) in enumerate(zip(elem['words'], elem['head_tags_gt'], elem['head_indices_gt'])) ]
    parser_labeled_results = {}
    parser_labeled_results['P'], parser_labeled_results['R'], parser_labeled_results['F1'] = filter_get_P_R_F1(parser_labeled_gt, parser_labeled_pred, ignore_list = ignore_edge_indices)

    ## parser unlabeled output
    parser_unlabeled_pred = [ f'{i}-{j}-{word}-{head_pred}' for i, elem in enumerate(model_output) for j, (word, head_pred) in enumerate(zip(elem['words'], elem['head_indices_pred']))]
    parser_unlabeled_gt = [ f'{i}-{j}-{word}-{head_gt}' for i, elem in enumerate(model_output) for j, (word, head_gt) in enumerate(zip(elem['words'], elem['head_indices_gt']))]
    parser_unlabeled_results = {}
    parser_unlabeled_results['P'], parser_unlabeled_results['R'], parser_unlabeled_results['F1'] = filter_get_P_R_F1(parser_unlabeled_gt, parser_unlabeled_pred, ignore_list = ignore_head_index)

    return {'tagger_results' :  tagger_results, 'parser_labeled_results': parser_labeled_results, 'parser_unlabeled_results' : parser_unlabeled_results}
