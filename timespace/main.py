import time
from config import *
from input import *
from alternating import *

if __name__ == "__main__":
    logging.info("Reading data......")
    g_ReadInputData()
    time_start = time.time()

    admm()

    time_end = time.time()
    f = open("output_path.csv", "w")
    f.write("iteration,vehicle_id,path_no_eq,path_time_sq\n")
    for i in range(g_number_of_ADMM_iterations):
        for v in range(g_number_of_vehicles - 1):
            f.write(str(i) + ",")
            f.write(str(v) + ",")
            str1 = ""
            str2 = ""
            for s in range(len(path_no_seq[i][v])):
                str1 = str1 + str(path_no_seq[i][v][s]) + "_"
                str2 = str2 + str(path_time_seq[i][v][s]) + "_"
            f.write((str1) + "," + (str2) + "\n")
    f.close()

    f = open("output_profit.csv", "w")
    f.write("iteration,")
    for n in range(1, g_number_of_customers + 1):
        f.write("%d ," % (n))
    f.write("\n")
    for i in range(g_number_of_ADMM_iterations):
        f.write(str(i) + ",")
        for n in range(g_number_of_customers):
            f.write(str(record_profit[i][n]) + ",")
    f.write("\n")

    f = open("output_gap.csv", "w")
    f.write("iteration,LB,UB,Repeated services,missed services \n")
    for i in range(g_number_of_ADMM_iterations):
        f.write(str(i) + ",")
        f.write(str(ADMM_local_lowerbound[i]) + ",")
        f.write(str(ADMM_local_upperbound[i]) + ",")
        for j in repeat_served[i]:
            f.write(str(j) + ";")
        for k in un_served[i]:
            f.write(str(k) + ";")
        f.write("\n")

    f.close()
    logging.info(f"finished {time_end - time_start}")
