/* file minunit_example.c */

#include <stdio.h>
#include "minunit.h"

#include "particle.h"

int tests_run = 0;
int tests_passed = 0;

static bool test_get_n_landmarks() {
    float particle[] = {
        1.0,
        2.0,
        0.1,
        0.9,
        100.0
    };

    mu_assert(get_n_landmarks(particle) == 100, "n_landmarks != 100");
    return true;
}

static bool test_get_mean() {
    float particle[] = {
        1.0,
        2.0,
        0.1,
        0.9,
        2.0,
        5.0, 6.0,
        3.0, 7.0
    };

    float *mean;
    mean = get_mean(particle, 0);
    mu_assert(mean[0] == 5.0 && mean[1] == 6.0, "mean != [5.0, 6.0]");

    mean = get_mean(particle, 1);
    mu_assert(mean[0] == 3.0 && mean[1] == 7.0, "mean != [3.0, 7.0]");

    return true;
}

static bool test_get_cov() {
    float particle[] = {
        1.0,
        2.0,
        0.1,
        0.9,
        2.0,
        5.0, 6.0,
        3.0, 7.0,
        1.0, 0, 0, 1.0,
        20.0, 0, 0, 20.0
    };

    float *cov;
    cov = get_cov(particle, 0);

    mu_assert(cov[0] == 1.0 && cov[1] == 0 && cov[2] == 0 && cov[3] == 1.0, "cov != [[1.0, 0], [0, 1.0]]");

    cov = get_cov(particle, 1);
    mu_assert(cov[0] == 20.0 && cov[1] == 0 && cov[2] == 0 && cov[3] == 20.0, "cov != [[20.0, 0], [0, 20.0]]");

    return true;
}

static void all_tests() {
    mu_run_test(test_get_n_landmarks);
    mu_run_test(test_get_mean);
    mu_run_test(test_get_cov);
}

int main(int argc, char **argv) {
    all_tests();
    printf("========================\n");
    printf("Tests results: %d/%d\n", tests_passed, tests_run);

    return tests_passed != tests_run;
    // return 0;
}