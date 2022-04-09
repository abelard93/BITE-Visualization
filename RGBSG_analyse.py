import numpy as np
from analyse_utils import *
from RGBSG_utils import load_RGBSG
import pickle

if __name__ == '__main__':

    method = 'BITES'
    results_dir='example_results/'
    compare_against_ATE = False

    X_train, Y_train, event_train, treatment_train, _, _ = load_RGBSG(partition='train', filename_="data/rgbsg.h5")
    X_test, Y_test, event_test, treatment_test, _, _ = load_RGBSG(partition='test', filename_="data/rgbsg.h5")

    result_path=results_dir + method + "_RGBSG"
        
    if method == 'BITES':
        model, config = get_best_model(result_path)
        model.compute_baseline_hazards(X_train, [Y_train, event_train, treatment_train])

        C_index, C_index_T0, C_index_T1 = get_C_Index_BITES(model, X_test, Y_test, event_test, treatment_test)
        pred_ite, _ = get_ITE_BITES(model, X_test, treatment_test)

        if compare_against_ATE:
            analyse_randomized_test_set(np.ones_like(pred_ite), Y_test, event_test, treatment_test, C_index=C_index,
                                        method_name=None, save_path=None, annotate=False)
            analyse_randomized_test_set(pred_ite, Y_test, event_test, treatment_test, C_index=C_index, method_name=method,
            							save_path='RGBSG_' + method + '_baseline.pdf', new_figure=False, annotate=True)
        else:
            analyse_randomized_test_set(pred_ite, Y_test, event_test, treatment_test, C_index=C_index,
                                        method_name=method, save_path='RGBSG_' + method + '.pdf')

        print(model)
        fileout = open("pickout.pkl", "wb")
        pickle.dump(model, fileout)
        fileout.close()

