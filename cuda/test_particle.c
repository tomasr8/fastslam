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
        100.0,
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

static bool test_add_landmark() {
    float particle[] = {
        1.0,
        2.0,
        0.1,
        0.9,
        3.0,
        2.0,
        5.0, 6.0,
        3.0, 7.0,
        0, 0,
        1.0, 0, 0, 1.0,
        20.0, 0, 0, 20.0,
        0, 0, 0, 0
    };

    float mean[] = { 42.0, 101.0 };
    float cov[] = { 5.0, 0, 0, 5.0 };

    add_landmark(particle, mean, cov);

    mu_assert(get_n_landmarks(particle) == 3, "n_landmarks != 3");
    float *new_mean = get_mean(particle, 2);
    float *new_cov = get_cov(particle, 2);

    mu_assert(new_mean[0] == 42.0 && new_mean[1] == 101.0, "mean != [42.0, 101.0]");
    mu_assert(new_cov[0] == 5.0 && new_cov[1] == 0 && new_cov[2] == 0 && new_cov[3] == 5.0, "cov != [[5.0, 0], [0, 5.0]]");

    return true;
}

static bool test_add_unassigned_measurements_as_landmarks() {
    float particle[] = {
        1.0,
        2.0,
        0.1,
        0.9,
        4.0,
        2.0,
        5.0, 6.0,
        3.0, 7.0,
        0, 0,
        0, 0,
        1.0, 0, 0, 1.0,
        20.0, 0, 0, 20.0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };

    bool assigned_measurements[] = { false, true, false };
    int n_measurements = 3;
    // float **measurements = malloc(n_measurements * sizeof(float*));
    // for(int i = 0; i < n_measurements; i++) {
    //     measurements[i] = malloc(2 * sizeof(float));
    // }
    // measurements[0] = { 14.0, 15.0 };
    float measurements[3][2] = {
        { 14.0, 15.0 },
        { 16.0, 17.0 },
        { 18.0, 19.0 }
    };
    float cov[] = { 3.0, 0, 0, 3.0 };

    add_unassigned_measurements_as_landmarks(particle, assigned_measurements, measurements, n_measurements, cov);

    mu_assert(get_n_landmarks(particle) == 4, "n_landmarks != 4");
    float *new_mean, *new_cov;
    new_mean = get_mean(particle, 2);
    new_cov = get_cov(particle, 2);

    // printf("%f %f %f %f\n", new_cov[0], new_cov[1], new_cov[2], new_cov[3]);

    mu_assert(new_mean[0] == 15.0 && new_mean[1] == 17.0, "mean != [15.0, 17.0]");
    mu_assert(new_cov[0] == 3.0 && new_cov[1] == 0 && new_cov[2] == 0 && new_cov[3] == 3.0, "cov != [[3.0, 0], [0, 3.0]]");

    // float *new_cov = get_cov(particle, 2);

    // mu_assert(new_cov[0] == 5.0 && new_cov[1] == 0 && new_cov[2] == 0 && new_cov[3] == 5.0, "cov != [[5.0, 0], [0, 5.0]]");

    return true;
}

static void all_tests() {
    mu_run_test(test_get_n_landmarks);
    mu_run_test(test_get_mean);
    mu_run_test(test_get_cov);
    mu_run_test(test_add_landmark);
    mu_run_test(test_add_unassigned_measurements_as_landmarks);
}

int main(int argc, char **argv) {
    all_tests();
    printf("========================\n");
    printf("Tests results: %d/%d\n", tests_passed, tests_run);

    return tests_passed != tests_run;
    // return 0;
}