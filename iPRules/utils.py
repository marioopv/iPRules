
def writereport(route, avg_tr_dth, avg_te_dth, avg_tr_srv, avg_te_srv, prec, rec, acc):
    """
    Escribe los coeficientes de predicción del modelo en un fichero.txt, además de los valores de precission, recall y accuracy del modelo de esa iteración.
    :param route:
    :param avg_tr_dth:
    :param avg_te_dth:
    :param avg_tr_srv:
    :param avg_te_srv:
    :param prec:
    :param rec:
    :param acc:
    :return:
    """
    comb_header = 'Pathologies combinations results:\n'
    death_line = 'Correct predicted casualties in training:' + str(
        avg_tr_dth) + ' || Correct predicted casualties in test:' + str(avg_te_dth) + '\n'
    surv_line = 'Correct predicted csurvivors in training:' + str(
        avg_tr_srv) + ' || Correct predicted survivors in test:' + str(avg_te_srv) + '\n'
    model_header = 'Prediction model results:\n'
    prec_line = "Precission Test: " + str(prec) + '\n'
    rec_line = "Recall Test: " + str(rec) + '\n'
    acc_line = "Accuracy Test: " + str(acc) + '\n'
    f = open(route, 'w')
    f.write(comb_header)
    f.write(death_line)
    f.write(surv_line)
    f.write(model_header)
    f.write(prec_line)
    f.write(rec_line)
    f.write(acc_line)
    f.close()
