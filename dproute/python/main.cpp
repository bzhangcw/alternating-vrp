#include "dp.h"

int main(int argc, char *argv[]) {
    using namespace std;
    bool bool_speed_option = (string(argv[3]) == "T");
    bool bool_batch_option = (string(argv[2]) == "T");
    run_test(argv[1], bool_batch_option, bool_speed_option);
    return 1;
}
