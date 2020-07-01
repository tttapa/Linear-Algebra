#include <gtest/gtest.h>

#include <linalg/PermutationMatrix.hpp>

static PermutationMatrix::Permutation long_permutation = {
    544, 387,  2,    383, 807, 707,  34,   681,  109,  268, 570,  989,  491,
    906, 967,  331,  867, 944, 634,  229,  179,  982,  216, 627,  578,  296,
    343, 395,  895,  795, 196, 999,  238,  588,  110,  480, 961,  693,  708,
    581, 87,   731,  613, 278, 679,  27,   640,  535,  658, 800,  236,  105,
    934, 260,  719,  691, 617, 455,  918,  987,  375,  344, 239,  677,  732,
    840, 26,   81,   829, 54,  698,  125,  607,  1023, 254, 560,  47,   62,
    176, 751,  759,  984, 958, 733,  697,  527,  358,  410, 773,  806,  684,
    467, 337,  100,  241, 883, 816,  200,  655,  769,  687, 667,  724,  775,
    354, 608,  453,  879, 271, 56,   206,  1000, 325,  812, 702,  454,  786,
    739, 163,  95,   528, 776, 171,  740,  645,  441,  423, 898,  311,  571,
    584, 83,   596,  866, 804, 814,  725,  293,  31,   805, 101,  737,  255,
    626, 674,  815,  71,  529, 204,  996,  3,    465,  817, 609,  943,  75,
    70,  586,  534,  4,   245, 477,  767,  50,   399,  891, 360,  846,  519,
    304, 848,  334,  36,  910, 886,  40,   834,  610,  270, 513,  231,  818,
    111, 837,  153,  295, 256, 885,  661,  396,  952,  641, 624,  656,  292,
    335, 960,  904,  68,  998, 226,  121,  84,   946,  671, 827,  159,  788,
    408, 119,  587,  532, 300, 308,  402,  464,  583,  151, 819,  12,   683,
    495, 215,  273,  520, 585, 761,  605,  758,  146,  601, 420,  662,  575,
    828, 882,  675,  505, 822, 997,  452,  170,  956,  377, 939,  152,  313,
    154, 979,  551,  917, 764, 48,   149,  189,  106,  747, 422,  184,  771,
    82,  227,  85,   874, 370, 98,   678,  482,  404,  9,   1012, 558,  977,
    823, 13,   316,  7,   376, 905,  434,  5,    915,  838, 164,  64,   329,
    201, 479,  167,  177, 42,  69,   439,  319,  954,  789, 359,  406,  280,
    494, 600,  794,  976, 361, 590,  21,   409,  650,  715, 372,  868,  317,
    778, 266,  705,  770, 853, 1013, 630,  117,  172,  690, 692,  858,  673,
    147, 629,  286,  131, 685, 156,  38,   333,  833,  801, 821,  926,  820,
    869, 440,  525,  173, 345, 392,  353,  900,  508,  742, 643,  492,  689,
    631, 213,  857,  202, 660, 1010, 35,   721,  92,   704, 275,  400,  323,
    380, 963,  476,  638, 11,  429,  140,  851,  78,   16,  107,  1003, 937,
    485, 861,  657,  580, 458, 752,  194,  757,  122,  301, 753,  94,   948,
    272, 957,  307,  504, 137, 24,   847,  972,  965,  314, 385,  942,  762,
    97,  924,  257,  499, 472, 487,  193,  793,  718,  501, 839,  625,  902,
    261, 63,   80,   844, 842, 99,   456,  774,  521,  348, 165,  1004, 1016,
    651, 903,  431,  263, 604, 86,   284,  923,  945,  860, 281,  336,  808,
    357, 174,  493,  864, 389, 332,  414,  309,  865,  797, 497,  158,  66,
    267, 252,  781,  784, 155, 129,  510,  162,  914,  362, 680,  490,  483,
    55,  933,  595,  639, 130, 457,  425,  46,   115,  978, 798,  859,  251,
    686, 364,  873,  426, 185, 302,  390,  388,  881,  652, 632,  665,  33,
    148, 339,  235,  394, 15,  398,  444,  549,  219,  310, 573,  766,  471,
    727, 128,  791,  182, 190, 91,   582,  274,  230,  597, 90,   208,  533,
    531, 552,  593,  218, 234, 694,  930,  175,  132,  133, 1022, 79,   623,
    516, 889,  366,  247, 475, 188,  240,  659,  450,  899, 468,  711,  262,
    363, 548,  1015, 39,  975, 469,  854,  28,   346,  772, 811,  352,  139,
    701, 526,  992,  983, 594, 52,   591,  289,  647,  437, 893,  448,  446,
    901, 413,  187,  116, 58,  282,  726,  19,   74,   143, 763,  1020, 576,
    826, 717,  832,  862, 636, 537,  297,  670,  602,  25,  32,   951,  612,
    554, 104,  974,  411, 225, 913,  183,  73,   569,  567, 436,  540,  315,
    745, 1019, 1021, 324, 18,  557,  478,  145,  518,  312, 330,  166,  672,
    856, 755,  338,  41,  384, 1017, 168,  710,  779,  941, 424,  401,  722,
    799, 221,  355,  546, 435, 579,  14,   1001, 318,  973, 222,  845,  611,
    77,  144,  835,  191, 123, 995,  561,  966,  460,  265, 577,  696,  936,
    813, 589,  931,  418, 368, 342,  445,  959,  603,  741, 765,  922,  136,
    415, 306,  443,  447, 416, 114,  633,  249,  112,  940, 870,  908,  207,
    743, 43,   347,  785, 809, 264,  246,  964,  562,  427, 430,  1011, 642,
    887, 288,  559,  291, 378, 709,  299,  920,  509,  473, 373,  796,  382,
    664, 371,  1007, 530, 417, 192,  203,  894,  1014, 880, 237,  412,  488,
    736, 990,  841,  754, 852, 120,  484,  44,   716,  782, 374,  592,  340,
    466, 803,  539,  290, 214, 341,  287,  890,  825,  489, 831,  648,  713,
    421, 169,  233,  210, 921, 350,  449,  205,  397,  244, 738,  810,  962,
    947, 615,  712,  563, 322, 181,  113,  950,  730,  748, 669,  486,  616,
    279, 403,  556,  746, 877, 723,  474,  863,  878,  217, 118,  668,  970,
    706, 45,   897,  522, 744, 1018, 916,  23,   481,  503, 22,   783,  849,
    876, 695,  459,  875, 653, 407,  1009, 919,  512,  792, 566,  500,  391,
    428, 760,  461,  802, 59,  127,  699,  356,  209,  29,  720,  124,  96,
    381, 714,  768,  991, 777, 393,  622,  928,  756,  524, 506,  37,   912,
    135, 199,  790,  734, 729, 65,   250,  969,  178,  550, 985,  259,  980,
    224, 502,  896,  138, 971, 8,    927,  541,  925,  60,  243,  134,  328,
    242, 654,  349,  108, 496, 884,  320,  253,  523,  703, 51,   17,   258,
    855, 433,  511,  6,   949, 432,  635,  150,  379,  574, 555,  536,  929,
    598, 76,   637,  305, 621, 223,  618,  470,  514,  572, 545,  935,  750,
    61,  72,   438,  850, 599, 365,  142,  542,  327,  298, 517,  197,  993,
    228, 498,  728,  564, 0,   212,  824,  888,  676,  682, 644,  994,  93,
    553, 88,   211,  220, 160, 700,  663,  326,  968,  67,  666,  198,  49,
    830, 386,  20,   405, 277, 911,  157,  907,  955,  568, 180,  10,   126,
    538, 614,  909,  462, 787, 30,   619,  872,  141,  628, 988,  1002, 442,
    102, 283,  735,  367, 89,  419,  1006, 606,  507,  932, 276,  892,  103,
    871, 953,  369,  1,   57,  186,  780,  351,  749,  463, 248,  232,  161,
    303, 565,  321,  843, 285, 269,  1005, 620,  981,  836, 195,  1008, 646,
    543, 986,  515,  547, 451, 688,  53,   938,  294,  649,
};

TEST(PermutationMatrix, permutationConversion) {
    PermutationMatrix P = PermutationMatrix::from_permutation(long_permutation);
    PermutationMatrix::Permutation result = P.to_permutation();
    EXPECT_EQ(result, long_permutation);
}

TEST(PermutationMatrix, permutationConversionInverse) {
    PermutationMatrix P = PermutationMatrix::from_permutation(long_permutation);
    PermutationMatrix::Permutation result = transpose(P).to_permutation();

    PermutationMatrix::Permutation expected(long_permutation.size());
    for (size_t i = 0; i < long_permutation.size(); ++i)
        expected[long_permutation[i]] = i;

    EXPECT_EQ(result, expected);
}

TEST(PermutationMatrix, random) {
    PermutationMatrix P = PermutationMatrix::random(128);
    PermutationMatrix::Permutation identity =
        PermutationMatrix::identity_permutation(128);
    PermutationMatrix::Permutation result = P.to_permutation();
    EXPECT_TRUE(
        std::is_permutation(result.begin(), result.end(), identity.begin()));
}

TEST(PermutationMatrix, MATLABConvention) {
    // This tests the MATLAB permutation index vector compatibility.
    // Equivalent matlab code: result = A(P,:)
    // (after accounting for 1-based vs 0-based array indexing)
    Matrix A = {
        {11, 12, 13, 14, 15, 16}, {21, 22, 23, 24, 25, 26},
        {31, 32, 33, 34, 35, 36}, {41, 42, 43, 44, 45, 56},
        {51, 52, 53, 54, 55, 56}, {61, 62, 63, 64, 65, 66},
    };
    PermutationMatrix P = {
        {4, 2, 0, 5, 3, 1},
        PermutationMatrix::RowPermutation,
    };
    Matrix expected = {
        {51, 52, 53, 54, 55, 56}, {31, 32, 33, 34, 35, 36},
        {11, 12, 13, 14, 15, 16}, {61, 62, 63, 64, 65, 66},
        {41, 42, 43, 44, 45, 56}, {21, 22, 23, 24, 25, 26},
    };
    Matrix result = P * A;
    EXPECT_EQ(result, expected);
}

// Matrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(PermutationMatrix, permutationLeft) {
    Matrix A = {
        {11, 12, 13},
        {21, 22, 23},
        {31, 32, 33},
        {41, 42, 43},
    };
    PermutationMatrix P = {{1, 3, 2, 0}, PermutationMatrix::RowPermutation};
    Matrix expected = {
        {21, 22, 23},
        {41, 42, 43},
        {31, 32, 33},
        {11, 12, 13},
    };
    Matrix result = P * A;
    EXPECT_EQ(result, expected);
}

TEST(PermutationMatrix, permutationRight) {
    Matrix A = {
        {11, 12, 13, 14},
        {21, 22, 23, 24},
        {31, 32, 33, 34},
    };
    PermutationMatrix P = {{1, 3, 2, 0}, PermutationMatrix::ColumnPermutation};
    Matrix expected = {
        {12, 14, 13, 11},
        {22, 24, 23, 21},
        {32, 34, 33, 31},
    };
    Matrix result = A * P;
    EXPECT_EQ(result, expected);
}

TEST(PermutationMatrix, permutationLeftInverse) {
    Matrix A = {
        {11, 12, 13},
        {21, 22, 23},
        {31, 32, 33},
        {41, 42, 43},
    };
    PermutationMatrix P = {{1, 3, 2, 0}, PermutationMatrix::RowPermutation};
    Matrix expected = {
        {41, 42, 43},
        {11, 12, 13},
        {31, 32, 33},
        {21, 22, 23},
    };
    Matrix result = transpose(P) * A;
    EXPECT_EQ(result, expected);
}

TEST(PermutationMatrix, permutationRightInverse) {
    Matrix A = {
        {11, 12, 13, 14},
        {21, 22, 23, 24},
        {31, 32, 33, 34},
    };
    PermutationMatrix P = {{1, 3, 2, 0}, PermutationMatrix::ColumnPermutation};
    Matrix expected = {
        {14, 11, 13, 12},
        {24, 21, 23, 22},
        {34, 31, 33, 32},
    };
    Matrix result = A * transpose(P);
    EXPECT_EQ(result, expected);
}

// SquareMatrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(PermutationMatrix, permutationLeftSquareMatrix) {
    SquareMatrix A = {
        {11, 12, 13, 14},
        {21, 22, 23, 24},
        {31, 32, 33, 34},
        {41, 42, 43, 44},
    };
    PermutationMatrix P = {{1, 3, 2, 0}, PermutationMatrix::RowPermutation};
    SquareMatrix expected = {
        {21, 22, 23, 24},
        {41, 42, 43, 44},
        {31, 32, 33, 34},
        {11, 12, 13, 14},
    };
    SquareMatrix result = P * A;
    EXPECT_EQ(result, expected);
}

TEST(PermutationMatrix, permutationRightSquareMatrix) {
    SquareMatrix A = {
        {11, 12, 13, 14}, {21, 22, 23, 24}, {31, 32, 33, 34}, {41, 42, 43, 44}};
    PermutationMatrix P = {{1, 3, 2, 0}, PermutationMatrix::ColumnPermutation};
    SquareMatrix expected = {
        {12, 14, 13, 11},
        {22, 24, 23, 21},
        {32, 34, 33, 31},
        {42, 44, 43, 41},
    };
    SquareMatrix result = A * P;
    EXPECT_EQ(result, expected);
}

// Vector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(PermutationMatrix, permutationLeftVector) {
    Vector A = {1, 2, 3, 4};
    PermutationMatrix P = {{1, 3, 2, 0}, PermutationMatrix::RowPermutation};
    Vector expected = {2, 4, 3, 1};
    Vector result = P * A;
    EXPECT_EQ(result, expected);
}

// RowVector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(PermutationMatrix, permutationRightVector) {
    RowVector A = {1, 2, 3, 4};
    PermutationMatrix P = {{1, 3, 2, 0}, PermutationMatrix::ColumnPermutation};
    RowVector expected = {2, 4, 3, 1};
    RowVector result = A * P;
    EXPECT_EQ(result, expected);
}

// Matrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(PermutationMatrix, permutationLeftMove) {
    Matrix A = {
        {11, 12, 13},
        {21, 22, 23},
        {31, 32, 33},
        {41, 42, 43},
    };
    PermutationMatrix P = {{1, 3, 2, 0}, PermutationMatrix::RowPermutation};
    Matrix expected = {
        {21, 22, 23},
        {41, 42, 43},
        {31, 32, 33},
        {11, 12, 13},
    };
    Matrix result = P * std::move(A);
    EXPECT_EQ(result, expected);
}

TEST(PermutationMatrix, permutationRightMove) {
    Matrix A = {
        {11, 12, 13, 14},
        {21, 22, 23, 24},
        {31, 32, 33, 34},
    };
    PermutationMatrix P = {{1, 3, 2, 0}, PermutationMatrix::ColumnPermutation};
    Matrix expected = {
        {12, 14, 13, 11},
        {22, 24, 23, 21},
        {32, 34, 33, 31},
    };
    Matrix result = std::move(A) * P;
    EXPECT_EQ(result, expected);
}

// SquareMatrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(PermutationMatrix, permutationLeftSquareMatrixMove) {
    SquareMatrix A = {
        {11, 12, 13, 14},
        {21, 22, 23, 24},
        {31, 32, 33, 34},
        {41, 42, 43, 44},
    };
    PermutationMatrix P = {{1, 3, 2, 0}, PermutationMatrix::RowPermutation};
    SquareMatrix expected = {
        {21, 22, 23, 24},
        {41, 42, 43, 44},
        {31, 32, 33, 34},
        {11, 12, 13, 14},
    };
    SquareMatrix result = P * std::move(A);
    EXPECT_EQ(result, expected);
}

TEST(PermutationMatrix, permutationRightSquareMatrixMove) {
    SquareMatrix A = {
        {11, 12, 13, 14}, {21, 22, 23, 24}, {31, 32, 33, 34}, {41, 42, 43, 44}};
    PermutationMatrix P = {{1, 3, 2, 0}, PermutationMatrix::ColumnPermutation};
    SquareMatrix expected = {
        {12, 14, 13, 11},
        {22, 24, 23, 21},
        {32, 34, 33, 31},
        {42, 44, 43, 41},
    };
    SquareMatrix result = std::move(A) * P;
    EXPECT_EQ(result, expected);
}

// Vector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(PermutationMatrix, permutationLeftVectorMove) {
    Vector A = {1, 2, 3, 4};
    PermutationMatrix P = {{1, 3, 2, 0}, PermutationMatrix::RowPermutation};
    Vector expected = {2, 4, 3, 1};
    Vector result = P * std::move(A);
    EXPECT_EQ(result, expected);
}

// RowVector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(PermutationMatrix, permutationRightVectorMove) {
    RowVector A = {1, 2, 3, 4};
    PermutationMatrix P = {{1, 3, 2, 0}, PermutationMatrix::ColumnPermutation};
    RowVector expected = {2, 4, 3, 1};
    RowVector result = std::move(A) * P;
    EXPECT_EQ(result, expected);
}