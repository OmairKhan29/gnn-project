import json, numpy as np 
 
with open('results/ablations/ablation_strategy.json', 'r') as f: 
    data = json.load(f) 
 
seed1_result = { 
    'best_val_auc': 0.7849008769465196, 
    'test_auc_avg': 0.7089616551230404, 
    'test_auc_per_task': {'bbbp_p_np': 0.6657192407746411, 'bace_Class': 0.6938405797101449, 'hiv_HIV_active': 0.7439767414905536, 'clintox_FDA_APPROVED': 0.7553956834532374, 'clintox_CT_TOX': 0.7380434782608696, 'tox21_NR-AR': 0.7985498726239467, 'tox21_NR-AR-LBD': 0.7372554422705979, 'tox21_NR-AhR': 0.8169319826338639, 'tox21_NR-Aromatase': 0.6496588806660499, 'tox21_NR-ER': 0.5610298537367262, 'tox21_NR-ER-LBD': 0.6474815648043994, 'tox21_NR-PPAR-gamma': 0.6386931071556565, 'tox21_SR-ARE': 0.6198547215496367, 'tox21_SR-ATAD5': 0.7288484848484849, 'tox21_SR-HSE': 0.7567645698427382, 'tox21_SR-MMP': 0.7595827268316352, 'tox21_SR-p53': 0.740721206438506}, 
    'num_params': 757201, 'training_time': 0, 'epochs_trained': 89 
} 
 
seed2_result = { 
    'best_val_auc': 0.7907770270318006, 
    'test_auc_avg': 0.7192058521757341, 
    'test_auc_per_task': {'bbbp_p_np': 0.6357067154831871, 'bace_Class': 0.7840579710144928, 'hiv_HIV_active': 0.7225572769771664, 'clintox_FDA_APPROVED': 0.7170263788968825, 'clintox_CT_TOX': 0.7753623188405797, 'tox21_NR-AR': 0.7768469527728786, 'tox21_NR-AR-LBD': 0.7922981537613667, 'tox21_NR-AhR': 0.8167274900899767, 'tox21_NR-Aromatase': 0.6873843663274746, 'tox21_NR-ER': 0.5764375876577841, 'tox21_NR-ER-LBD': 0.6464191976003, 'tox21_NR-PPAR-gamma': 0.6741727392187312, 'tox21_SR-ARE': 0.6363960749330955, 'tox21_SR-ATAD5': 0.7428686868686869, 'tox21_SR-HSE': 0.7421080018501387, 'tox21_SR-MMP': 0.7706210577389616, 'tox21_SR-p53': 0.7295085169557743}, 
    'num_params': 757201, 'training_time': 0, 'epochs_trained': 126 
} 
 
data['results']['task_conditioned_pcgrad']['per_seed'].append(seed1_result) 
data['results']['task_conditioned_pcgrad']['per_seed'].append(seed2_result) 
 
all_aucs = [r['test_auc_avg'] for r in data['results']['task_conditioned_pcgrad']['per_seed']] 
data['results']['task_conditioned_pcgrad']['mean_auc'] = float(np.mean(all_aucs)) 
data['results']['task_conditioned_pcgrad']['std_auc'] = float(np.std(all_aucs)) 
 
with open('results/ablations/ablation_strategy.json', 'w') as f: 
    json.dump(data, f, indent=2) 
 
print('Updated! New PCGrad stats:') 
print(f'  Seeds: {len(all_aucs)}') 
print(f'  Mean AUC: {np.mean(all_aucs):.4f}') 
print(f'  Std AUC: {np.std(all_aucs):.4f}') 
