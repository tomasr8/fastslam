#include <stdio.h>
#include <stdbool.h>

#define mu_assert(test, message) do { if (!(test)) { printf("\033[0;31m%s\033[0m  %s\n", __func__, message); return false; } else { printf("\033[0;32m%s\033[0m  ✓\n", __func__); } } while (0)
#define mu_run_test(test) do { bool result = test(); tests_run++; if (result) { tests_passed++; } } while (0)

extern int tests_run;
extern int tests_passed;
