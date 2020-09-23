#include "sort.h"

// A utility function to swap two elements
void swap(float *a, float *b)
{
    float t = *a;
    *a = *b;
    *b = t;
}

void swap_idx(int *a, int *b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

/* This function takes last element as pivot, places 
   the pivot element at its correct position in sorted 
    array, and places all smaller (smaller than pivot) 
   to left of pivot and all greater elements to right 
   of pivot */
int partition(float arr[], int lm[], int me[], int low, int high)
{
    float pivot = arr[high]; // pivot
    int i = (low - 1);       // Index of smaller element

    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than the pivot
        if (arr[j] < pivot)
        {
            i++; // increment index of smaller element
            swap(&arr[i], &arr[j]);
            swap_idx(&lm[i], &lm[j]);
            swap_idx(&me[i], &me[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    swap_idx(&lm[i + 1], &lm[high]);
    swap_idx(&me[i + 1], &me[high]);
    return (i + 1);
}

/* The main function that implements quicksort 
 arr[] --> Array to be sorted, 
  low  --> Starting index, 
  high  --> Ending index */
void quicksort(float arr[], int lm[], int me[], int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now 
           at right place */
        int pi = partition(arr, lm, me, low, high);

        // Separately sort elements before
        // partition and after partition
        quicksort(arr, lm, me, low, pi - 1);
        quicksort(arr, lm, me, pi + 1, high);
    }
}

void printArray(float arr[], int lm[], int me[], int size) 
{ 
    int i; 
    for (i=0; i < size; i++) 
        printf("%f ", arr[i]); 
    printf("\n");

    for (i=0; i < size; i++) 
        printf("%d ", lm[i]); 
    printf("\n"); 

    for (i=0; i < size; i++) 
        printf("%d ", me[i]); 
    printf("\n"); 
} 

// int main()
// {
//     float arr[] = {10.0, 7.0, 8.0, 9.0};
//     int lm[] = {0, 0, 1, 1};
//     int me[] = {0, 1, 0, 1};



//     int n = sizeof(arr) / sizeof(arr[0]);
//     quicksort(arr, lm, me, 0, n - 1);
//     printf("Sorted array: \n");
//     printArray(arr, lm, me, n);
//     return 0;
// }
