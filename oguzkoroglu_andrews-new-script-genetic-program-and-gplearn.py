# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/andrews-new-script-plus-a-genetic-program-model"))

print(os.listdir("../input/lanl-features"))

# Any results you write to the current directory are saved as output.
def GPIII(data):

    return (5.577521 +

            0.040000*np.tanh(((((((data["ffti_range_0_1000"]) - (((((data["min_roll_std_10000"]) + (((((data["ffti_range_3000_4000"]) + (data["num_peaks_10"]))) * 2.0)))) + (((data["percentile_roll_std_20"]) * 2.0)))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((data["num_peaks_10"]) - (((((data["iqr"]) + (((((((((data["num_peaks_10"]) + (data["percentile_roll_std_5"]))) + (data["ffti_range_3000_4000"]))) * 2.0)) * 2.0)))) * 2.0)))) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["ffti_range_0_1000"]) - (data["percentile_roll_std_1"]))) - (data["num_peaks_10"]))) * 2.0)) - (((data["fftr_range_2000_3000"]) + (data["percentile_roll_std_25"]))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["ffti_range_-1000_0"]) - (((data["ffti_range_2000_3000"]) + (((data["percentile_roll_std_5"]) + (data["percentile_95"]))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["fftr_av_change_abs_roll_mean_100"]) + (((data["range_0_1000"]) * 2.0)))) * 2.0)) - (((data["fftr_range_-3000_-2000"]) + (data["num_peaks_10"]))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((data["fftr_av_change_abs_roll_std_500"]) + (((data["range_0_1000"]) * 2.0)))) * 2.0)) - (data["fftr_range_-3000_-2000"]))) * 2.0)) * 2.0)) * 2.0)) - (data["min_roll_std_10000"]))) * 2.0)) +

            0.040000*np.tanh((((((((((((-1.0*((((data["percentile_95"]) + (((data["percentile_roll_std_20"]) + (data["ffti_range_3000_4000"])))))))) - (data["num_peaks_10"]))) * 2.0)) * 2.0)) - (data["min_roll_std_500"]))) * 2.0)) +

            0.040000*np.tanh((((((-1.0*((((data["min_roll_std_500"]) + (((data["fftr_range_-3000_-2000"]) + (((((((data["iqr"]) - (data["range_0_1000"]))) + (data["percentile_roll_std_5"]))) * 2.0))))))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((data["range_0_1000"]) - (data["iqr"]))) - (data["num_peaks_10"]))) - (data["min_roll_std_10000"]))) - (data["ffti_range_3000_4000"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((-1.0*(((((((((data["ffti_range_-4000_-3000"]) + (data["percentile_roll_std_20"]))/2.0)) * 2.0)) * 2.0))))) - (data["num_peaks_10"]))) * 2.0)) - (data["iqr"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["ffti_range_0_1000"]) - (((((data["percentile_roll_std_1"]) + (((data["ffti_range_2000_3000"]) + (data["iqr"]))))) + (data["num_peaks_10"]))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((-1.0*((((((data["iqr"]) + (((((data["percentile_roll_std_10"]) + (((data["num_peaks_10"]) + (((data["fftr_range_-3000_-2000"]) + (data["percentile_95"]))))))) * 2.0)))) * 2.0))))) * 2.0)) +

            0.040000*np.tanh((((10.45759582519531250)) * (((data["range_0_1000"]) - ((((data["fftr_max_first_10000"]) + (((((data["ffti_range_-4000_-3000"]) + (((data["fftr_max_first_10000"]) - (((data["range_0_1000"]) * 2.0)))))) * 2.0)))/2.0)))))) +

            0.040000*np.tanh(((((((data["range_0_1000"]) - (data["iqr"]))) - (((((data["fftr_range_2000_3000"]) + (((((data["ave_roll_mean_500"]) - (((data["range_0_1000"]) * 2.0)))) * 2.0)))) * 2.0)))) * 2.0)) +

            0.040000*np.tanh(((((-1.0) + (((((-1.0) - (((((data["percentile_roll_std_20"]) + (((data["gmean"]) - (data["range_0_1000"]))))) * 2.0)))) * 2.0)))) * 2.0)) +

            0.040000*np.tanh((((-1.0*((((((((data["iqr"]) + (data["fftr_range_-3000_-2000"]))) + (((data["min_roll_std_500"]) + (((((data["ffti_range_3000_4000"]) + (data["percentile_roll_std_10"]))) * 2.0)))))) * 2.0))))) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((data["ffti_range_-1000_0"]) - (data["num_peaks_10"]))) - (data["abs_percentile_75"]))) * 2.0)) - (data["min_roll_std_1000"]))) - (data["percentile_roll_std_1"]))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["ffti_percentile_30"]) - (((((data["num_peaks_10"]) + (((data["iqr"]) + (data["percentile_roll_std_10"]))))) * 2.0)))) - (data["fftr_range_2000_3000"]))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((-1.0*((((((((data["abs_percentile_80"]) + (((((((data["fftr_std_roll_mean_10"]) + (data["num_peaks_10"]))) + (((data["percentile_roll_std_10"]) + (data["fftr_range_-3000_-2000"]))))) * 2.0)))) * 2.0)) * 2.0))))) +

            0.040000*np.tanh(((((((((((((((data["range_0_1000"]) * 2.0)) - (((data["gmean"]) + (data["fftr_range_-3000_-2000"]))))) - (data["num_peaks_10"]))) * 2.0)) - (data["min_roll_std_500"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((((-1.0*((((data["num_peaks_10"]) + (data["percentile_roll_std_10"])))))) - (((data["iqr"]) * 2.0)))) - (data["range_-1000_0"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((data["range_0_1000"]) - ((((data["fftr_range_-3000_-2000"]) + (((((data["percentile_roll_std_5"]) + (((data["abs_percentile_75"]) * 2.0)))) + (data["iqr"]))))/2.0)))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((data["range_0_1000"]) - (((data["abs_percentile_75"]) + (((((((data["fftr_range_-4000_-3000"]) + (data["percentile_roll_std_5"]))) + (data["abs_percentile_80"]))) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((data["ffti_range_0_1000"]) - (((data["min_roll_std_10000"]) + (data["min_roll_std_500"]))))) - (((data["fftr_range_-3000_-2000"]) + (((data["percentile_roll_std_5"]) + (data["abs_percentile_80"]))))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((((((((((-1.0*((data["num_peaks_10"])))) * 2.0)) - (data["percentile_roll_std_1"]))) * 2.0)) - (data["ffti_range_-4000_-3000"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((-1.0*((((data["percentile_roll_std_20"]) + (((((((data["fftr_range_2000_3000"]) + (((data["abs_percentile_75"]) + (data["num_peaks_10"]))))) * 2.0)) * 2.0))))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["percentile_roll_std_10"]) - (((((((data["percentile_roll_std_10"]) + (data["ffti_range_-4000_-3000"]))) + (data["abs_percentile_80"]))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((-1.0*((((data["iqr"]) * 2.0))))) - (((((data["num_peaks_10"]) - (data["mean_change_rate"]))) * 2.0)))) * 2.0)) - (data["percentile_roll_std_5"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((-2.0) - (((((data["iqr1"]) * 2.0)) + (((((data["percentile_roll_std_10"]) * 2.0)) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((((data["ffti_range_-1000_0"]) - (data["percentile_roll_std_10"]))) * 2.0)) * 2.0)) - (data["exp_Moving_average_3000_mean"]))) - (data["min_roll_std_50"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((data["range_0_1000"]) - (data["percentile_roll_std_10"]))) - (((data["abs_percentile_80"]) * 2.0)))) * 2.0)) * 2.0)) - (data["abs_percentile_75"]))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["num_peaks_10"]) - (((data["min_roll_std_500"]) + (((((((data["fftr_range_-3000_-2000"]) + (((data["num_peaks_10"]) + (data["range_-1000_0"]))))) * 2.0)) * 2.0)))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((data["ffti_range_-1000_0"]) - (((((((data["abs_percentile_80"]) + (data["fftr_range_2000_3000"]))) * 2.0)) + (((data["abs_percentile_80"]) + (data["percentile_roll_std_20"]))))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((-1.0*((((((data["min_roll_std_1000"]) * 2.0)) + (((((((data["min_roll_std_50"]) + (((((data["percentile_roll_std_10"]) + (data["ffti_range_-3000_-2000"]))) * 2.0)))) * 2.0)) * 2.0))))))) * 2.0)) +

            0.040000*np.tanh(((((((((data["percentile_5"]) - (((data["percentile_roll_std_10"]) + (data["abs_percentile_25"]))))) - (((((data["num_peaks_10"]) * 2.0)) + (data["iqr"]))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((((-1.0*((((((data["num_peaks_10"]) * 2.0)) + (((data["min_roll_std_1000"]) + (data["min_roll_std_100"])))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["fftr_min_roll_mean_100"]) - (((((data["num_peaks_10"]) * 2.0)) + (((((((data["percentile_roll_std_10"]) + (data["abs_percentile_80"]))) * 2.0)) + (data["abs_percentile_80"]))))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["mean_change_rate_first_50000"]) - (((data["min_roll_std_10000"]) + (((data["iqr"]) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) + (data["mean_change_rate_first_50000"]))) +

            0.040000*np.tanh((((((((-1.0*((((data["percentile_roll_std_5"]) - (data["range_0_1000"])))))) * 2.0)) - (((2.0) + (((data["percentile_roll_std_5"]) + (data["min_roll_std_100"]))))))) * 2.0)) +

            0.040000*np.tanh((((-1.0*((((((((((data["ffti_range_2000_3000"]) + (((data["abs_percentile_80"]) * 2.0)))) * 2.0)) + (((data["fftr_range_-4000_-3000"]) + (data["percentile_roll_std_10"]))))) * 2.0))))) * 2.0)) +

            0.040000*np.tanh(((((((((((data["abs_percentile_60"]) + (((((((((data["mean_change_rate"]) * 2.0)) - (data["percentile_roll_mean_40"]))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((-1.0*((data["num_peaks_10"])))) - (((((((((data["num_peaks_10"]) + (data["num_peaks_20"]))) + (((((data["abs_percentile_50"]) + (data["percentile_roll_std_1"]))) * 2.0)))) * 2.0)) * 2.0)))) +

            0.040000*np.tanh(((((((((((((((((data["mean_change_rate"]) - (((data["abs_percentile_90"]) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) - ((-1.0*((data["min_roll_std_500"])))))) * 2.0)) +

            0.040000*np.tanh((((((((((-1.0*((data["ffti_abs_percentile_30"])))) - (((data["ffti_max_roll_mean_10"]) + (((data["fftr_max"]) + (((data["percentile_roll_std_20"]) * 2.0)))))))) * 2.0)) - (data["min_roll_std_50"]))) * 2.0)) +

            0.040000*np.tanh(((((((((data["ffti_percentile_40"]) - (((data["min_roll_std_10000"]) + (((((data["iqr"]) + (((data["abs_percentile_50"]) + (data["iqr1"]))))) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["mean_change_rate_last_50000"]) + (((data["range_0_1000"]) * 2.0)))) - (((data["ffti_range_2000_3000"]) + (data["min_roll_std_500"]))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["mean_change_rate"]) * 2.0)) - (data["abs_percentile_25"]))) - (((data["min_roll_std_100"]) + (data["min_roll_std_1000"]))))) * 2.0)) - (data["abs_percentile_30"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["fftr_abs_trend"]) - (((data["min_roll_std_500"]) + (((data["percentile_90"]) + (data["num_peaks_10"]))))))) * 2.0)) - (data["num_peaks_10"]))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["abs_percentile_20"]) - (((((((((data["abs_percentile_50"]) - (data["range_0_1000"]))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["std_roll_mean_10000"]) - (((((data["num_peaks_10"]) - (data["mean_change_rate"]))) * 2.0)))) - (data["min_roll_std_10000"]))) * 2.0)) - (data["min_roll_std_100"]))) * 2.0)) +

            0.040000*np.tanh(((((((((((((((-2.0) * (data["percentile_95"]))) - ((((data["ffti_range_3000_4000"]) + (data["percentile_80"]))/2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((((data["mean_change_rate"]) * 2.0)) - (data["num_peaks_10"]))) * 2.0)) - (data["ffti_range_2000_3000"]))) * 2.0)) - (data["min_roll_std_500"]))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((((((((-1.0*((((data["min_roll_std_10000"]) + (data["abs_percentile_50"])))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) - (data["num_peaks_10"]))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((-3.0) + (((((((data["mean_change_rate_last_10000"]) * 2.0)) + (((((-3.0) - ((((9.0)) * (data["num_peaks_10"]))))) * 2.0)))) + (data["mean_change_rate_first_10000"]))))) +

            0.040000*np.tanh((((((((-1.0*((((data["abs_percentile_70"]) + (((((data["fftr_count_big"]) + (((data["percentile_roll_std_5"]) + (((data["percentile_80"]) * 2.0)))))) * 2.0))))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((-1.0*((((data["fftr_kurt"]) - ((((((-1.0*((((data["percentile_95"]) * 2.0))))) * 2.0)) * 2.0))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["mean_change_rate_last_50000"]) + (((((((((data["mean_change_rate_first_50000"]) + (((data["mean_change_rate_last_50000"]) - (data["num_peaks_10"]))))) * 2.0)) - (data["min_roll_std_100"]))) * 2.0)))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((9.0)) * ((((9.0)) * (((data["abs_percentile_20"]) - (((data["abs_percentile_25"]) + (((data["min_roll_std_10000"]) + (((data["percentile_90"]) * 2.0)))))))))))) +

            0.040000*np.tanh(((((((((((((((data["mean_change_rate_last_50000"]) + (data["mean_change_rate_first_50000"]))) - (data["num_peaks_10"]))) * 2.0)) - (((data["min_roll_std_100"]) - (data["mean_change_rate_last_10000"]))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((data["percentile_20"]) - (data["abs_percentile_70"]))) * 2.0)) - (((data["ffti_exp_Moving_std_300_mean"]) - (data["iqr"]))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((-1.0*((((data["abs_percentile_5"]) + ((((1.0)) - ((((-1.0*((data["num_peaks_10"])))) - (((data["percentile_90"]) * 2.0))))))))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((((-1.0) - (data["percentile_roll_std_10"]))) * 2.0)) * 2.0)) - (data["num_peaks_10"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) - (data["min_roll_std_100"]))) +

            0.040000*np.tanh(((((((((((((((((((data["iqr1"]) - (((data["percentile_90"]) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) - (data["percentile_90"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["ffti_trend"]) + (((data["percentile_20"]) - (((data["range_0_1000"]) - (((((data["percentile_20"]) - (data["percentile_80"]))) * 2.0)))))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["fftr_percentile_5"]) - ((((((data["percentile_80"]) + (((data["abs_percentile_70"]) * 2.0)))/2.0)) + (data["ffti_range_-3000_-2000"]))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((data["abs_percentile_90"]) + (((((data["abs_percentile_20"]) + (data["classic_sta_lta5_mean"]))) * 2.0)))/2.0)) - (((((data["percentile_90"]) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((-3.0) - ((-1.0*((((data["mean_change_rate_first_50000"]) - (((((data["percentile_90"]) + (((data["num_peaks_10"]) * 2.0)))) + (data["min_roll_std_100"])))))))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["mean_change_rate_first_10000"]) - (((data["min_roll_std_10000"]) + (data["min_roll_std_50"]))))) - (((data["fftr_percentile_roll_std_5"]) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["abs_percentile_20"]) - (((((((data["percentile_90"]) * 2.0)) * 2.0)) - (data["fftr_abs_percentile_60"]))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["mean_change_rate_last_10000"]) - (data["percentile_roll_mean_1"]))) + (((((data["mean_change_rate"]) + (data["mean_change_rate"]))) * 2.0)))) + (data["percentile_roll_std_75"]))) - (data["fftr_abs_percentile_10"]))) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["fftr_count_big"]) - (((data["percentile_90"]) * ((((data["fftr_percentile_roll_std_75"]) + ((5.0)))/2.0)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["ffti_mean_last_50000"]) * 2.0)) - (((data["fftr_abs_percentile_10"]) * 2.0)))) - (data["fftr_ave_roll_mean_10000"]))) - (data["num_peaks_100"]))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((((data["percentile_20"]) * 2.0)) - (data["percentile_80"]))) + (data["range_-1000_0"]))) * 2.0)) * 2.0)) * 2.0)) - (data["abs_percentile_5"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((data["num_peaks_10"]) - (((((data["percentile_roll_std_5"]) * (((data["num_peaks_10"]) * (((((data["min_roll_std_10000"]) + (data["num_peaks_10"]))) * 2.0)))))) + (data["classic_sta_lta7_mean"]))))) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((data["mean_change_rate"]) + (((-1.0) - (data["percentile_roll_mean_20"]))))) * 2.0)) + (data["mean_change_rate"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((data["abs_percentile_25"]) - (((((((((((data["num_peaks_10"]) + (data["abs_percentile_25"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) - (data["num_peaks_10"]))) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["abs_percentile_70"]) * (((data["percentile_20"]) * 2.0)))) - (data["ffti_range_2000_3000"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((-1.0*((((data["fftr_percentile_roll_mean_90"]) + (((data["num_peaks_10"]) - (((data["percentile_25"]) * ((13.67005634307861328))))))))))) * 2.0)) * 2.0)) * 2.0)) - (data["min_roll_std_1000"]))) +

            0.040000*np.tanh((-1.0*((((((((((data["percentile_80"]) + (((((((((data["ffti_spkt_welch_density_5"]) + (((data["fftr_percentile_roll_std_80"]) * 2.0)))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0))))) +

            0.040000*np.tanh(((((((data["abs_percentile_20"]) - (data["percentile_80"]))) - (data["min_roll_std_100"]))) - (((((data["fftr_Hann_window_mean_15000"]) - (((((data["ffti_abs_max_roll_mean_100"]) * (data["fftr_Hann_window_mean_15000"]))) * 2.0)))) * 2.0)))) +

            0.040000*np.tanh(((((((((((((((data["percentile_80"]) * (((data["iqr"]) * (((data["mean_change_rate"]) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) - (data["percentile_20"]))) * 2.0)) +

            0.040000*np.tanh(((((((((data["percentile_5"]) + (((((((data["ffti_percentile_30"]) + (((((data["classic_sta_lta6_mean"]) - (data["ave_roll_std_10"]))) * 2.0)))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["mean_change_rate_first_1000"]) - (data["min_roll_std_500"]))) + ((((-3.0) + (data["mean_change_rate"]))/2.0)))) * 2.0)) * 2.0)) + (data["mean_change_rate"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((data["percentile_80"]) * (data["percentile_roll_std_1"]))) + (data["percentile_80"]))) * 2.0)) * ((((((((((-1.0*((data["percentile_roll_std_1"])))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) +

            0.040000*np.tanh((((((((((((((-1.0*((data["ffti_percentile_roll_std_80"])))) * 2.0)) * 2.0)) - (((data["ffti_mean_first_50000"]) + ((((data["classic_sta_lta7_mean"]) + (data["min_roll_std_50"]))/2.0)))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["ffti_min_roll_std_10000"]) - (((data["fftr_min_roll_mean_10000"]) + (((((data["fftr_percentile_roll_mean_70"]) + (((((data["ffti_percentile_roll_std_80"]) + (data["ffti_percentile_roll_mean_50"]))) * 2.0)))) * 2.0)))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["fftr_abs_percentile_25"]) - (((data["fftr_range_0_1000"]) - (((((((data["range_0_1000"]) * 2.0)) * (((data["min_roll_std_10000"]) * (data["abs_percentile_90"]))))) * 2.0)))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((data["ffti_exp_Moving_average_50000_std"]) * (data["percentile_10"]))) - (((((data["autocorrelation_5"]) * (data["autocorrelation_5"]))) - (((data["ffti_Hann_window_mean_150"]) - (((data["fftr_exp_Moving_average_300_mean"]) - (data["mean_change_rate_first_1000"]))))))))) +

            0.040000*np.tanh(((((data["percentile_roll_std_80"]) + (data["num_peaks_10"]))) - (((((data["percentile_roll_std_10"]) * (((data["fftr_max_roll_mean_50"]) + (((data["num_peaks_10"]) * (data["num_peaks_10"]))))))) * 2.0)))) +

            0.040000*np.tanh(((data["mean_change_rate_first_50000"]) + (((((((data["percentile_10"]) * 2.0)) * 2.0)) * (((((data["sum"]) + (data["exp_Moving_average_50000_std"]))) + (((data["fftr_abs_percentile_10"]) + (data["exp_Moving_average_50000_std"]))))))))) +

            0.040000*np.tanh(((((((data["ffti_percentile_roll_mean_95"]) + (((((data["fftr_Hann_window_mean_1500"]) * (data["fftr_range_2000_3000"]))) - (((((((data["ffti_percentile_roll_std_80"]) * 2.0)) * 2.0)) * 2.0)))))) + (data["abs_percentile_50"]))) * 2.0)) +

            0.040000*np.tanh(((((data["abs_percentile_75"]) - (((data["fftr_range_1000_2000"]) + (((((data["abs_percentile_75"]) * (((data["percentile_roll_std_5"]) * (data["percentile_roll_std_25"]))))) * 2.0)))))) * 2.0)) +

            0.040000*np.tanh(((data["fftr_percentile_25"]) * (((((((((((((data["fftr_abs_percentile_60"]) - (((data["ffti_percentile_5"]) + (data["fftr_classic_sta_lta6_mean"]))))) * 2.0)) * 2.0)) + (data["exp_Moving_average_30000_std"]))) * 2.0)) * 2.0)))) +

            0.040000*np.tanh(((data["fftr_min_roll_std_500"]) - (((data["ffti_av_change_abs_roll_std_10"]) - (((((((((data["percentile_roll_std_10"]) * (data["num_peaks_10"]))) - (data["fftr_percentile_20"]))) * (data["fftr_percentile_20"]))) - (data["fftr_autocorrelation_50"]))))))) +

            0.040000*np.tanh(((data["percentile_10"]) + (((((data["percentile_10"]) * (((((data["autocorrelation_100"]) - (data["fftr_percentile_40"]))) - (((((data["fftr_exp_Moving_average_3000_std"]) * (data["classic_sta_lta7_mean"]))) * 2.0)))))) * 2.0)))) +

            0.040000*np.tanh(((((((((data["ffti_skew"]) - (((data["fftr_abs_percentile_30"]) * (((((data["min_roll_std_10000"]) - (data["fftr_percentile_roll_mean_10"]))) - (((data["fftr_percentile_25"]) * 2.0)))))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((data["skew"]) + (data["ffti_min_roll_mean_50"]))) - (((data["fftr_ave_roll_mean_50"]) + (((data["ffti_percentile_roll_mean_5"]) + ((-1.0*((((data["fftr_mean_first_50000"]) * (((data["fftr_range_2000_3000"]) * 2.0))))))))))))) +

            0.040000*np.tanh(((data["percentile_90"]) - (((((data["ffti_av_change_abs_roll_std_10000"]) + (((((((data["percentile_70"]) - (data["ffti_percentile_roll_std_50"]))) - (((data["ffti_percentile_roll_std_50"]) * (data["fftr_percentile_30"]))))) * 2.0)))) * 2.0)))) +

            0.040000*np.tanh(((((((data["exp_Moving_average_30000_std"]) + (((((data["percentile_20"]) + (((((data["classic_sta_lta7_mean"]) * (data["ffti_exp_Moving_average_50000_std"]))) * 2.0)))) * 2.0)))) + (data["autocorrelation_5"]))) + (data["mean_change_rate_first_1000"]))) +

            0.040000*np.tanh(((data["fftr_percentile_roll_std_50"]) + (((data["ffti_classic_sta_lta8_mean"]) * (((((((((((data["fftr_percentile_roll_mean_10"]) * 2.0)) - (((data["autocorrelation_100"]) + (data["ffti_percentile_roll_std_25"]))))) * 2.0)) * 2.0)) * 2.0)))))) +

            0.040000*np.tanh(((data["min_roll_std_500"]) - (((((((((((((((((data["fftr_percentile_roll_mean_1"]) + (data["ffti_spkt_welch_density_5"]))) * 2.0)) + (data["ffti_abs_std"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) +

            0.040000*np.tanh(((data["fftr_min_roll_std_500"]) + (((((data["exp_Moving_average_30000_mean"]) * (((-2.0) + ((-1.0*((((data["fftr_trend"]) * (((data["exp_Moving_average_30000_mean"]) - (data["ffti_max_roll_std_1000"])))))))))))) * 2.0)))) +

            0.040000*np.tanh(((((((((((data["percentile_60"]) * (data["percentile_20"]))) * 2.0)) + (((data["autocorrelation_50"]) - (data["num_peaks_50"]))))) * 2.0)) + (((data["autocorrelation_50"]) - (data["num_peaks_10"]))))) +

            0.040000*np.tanh(((data["ffti_max_to_min"]) - (((((data["ffti_min_roll_mean_100"]) * (((data["fftr_num_peaks_20"]) * 2.0)))) - (((data["autocorrelation_5"]) + ((((((data["autocorrelation_500"]) + (data["mean_change_rate_last_10000"]))/2.0)) * 2.0)))))))) +

            0.040000*np.tanh((((-1.0*((((((((((data["ffti_percentile_60"]) * 2.0)) * 2.0)) * 2.0)) * (data["fftr_min_roll_std_500"])))))) - (((data["autocorrelation_10"]) - (data["percentile_roll_std_60"]))))) +

            0.040000*np.tanh(((((data["mean_change_rate_first_1000"]) + (data["fftr_num_peaks_50"]))) - (((data["fftr_abs_max_roll_mean_100"]) - (((((data["fftr_min_roll_std_500"]) - (((data["classic_sta_lta2_mean"]) - (data["fftr_num_peaks_50"]))))) - (data["min_roll_std_50"]))))))) +

            0.040000*np.tanh(((data["fftr_mean_first_50000"]) + (((((data["classic_sta_lta5_mean"]) * 2.0)) - (((data["min_roll_std_10000"]) * (((((data["percentile_roll_std_10"]) * (((data["percentile_roll_std_10"]) + (data["fftr_mean_first_50000"]))))) * 2.0)))))))) +

            0.040000*np.tanh(((data["percentile_roll_std_70"]) - (((data["ffti_percentile_60"]) - (((data["fftr_abs_percentile_10"]) * (((((data["fftr_percentile_roll_mean_70"]) - (data["exp_Moving_average_50000_std"]))) - (((data["percentile_roll_std_70"]) * (data["fftr_percentile_70"]))))))))))) +

            0.040000*np.tanh(((((((((data["mean_change_rate_first_50000"]) * (((data["percentile_50"]) - (((data["fftr_range_0_1000"]) * (((data["percentile_75"]) + (data["fftr_percentile_roll_mean_10"]))))))))) * 2.0)) * 2.0)) - (data["fftr_percentile_roll_std_25"]))) +

            0.040000*np.tanh(((((data["ffti_percentile_roll_mean_95"]) + (((((((data["fftr_range_-1000_0"]) * (data["ffti_percentile_roll_std_50"]))) * 2.0)) * 2.0)))) + (((((data["ffti_percentile_roll_std_50"]) + (data["skew"]))) + (data["mean_change_rate_first_50000"]))))) +

            0.040000*np.tanh(((((((((((data["ffti_range_-3000_-2000"]) * (((data["ffti_percentile_roll_std_1"]) + (data["ffti_autocorrelation_10000"]))))) * 2.0)) + (data["percentile_roll_std_70"]))) - (((data["ffti_abs_max_roll_std_500"]) * (data["kurt"]))))) * 2.0)) +

            0.040000*np.tanh(((((((data["percentile_50"]) * (((data["fftr_range_2000_3000"]) - (((data["ave_roll_std_100"]) - ((((data["ffti_av_change_abs_roll_mean_10"]) + ((((data["range_0_1000"]) + (data["fftr_range_1000_2000"]))/2.0)))/2.0)))))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((data["num_peaks_10"]) + (((data["ffti_range_-3000_-2000"]) - (((((data["ffti_range_2000_3000"]) * 2.0)) * (((((data["num_peaks_10"]) * (data["num_peaks_10"]))) + (data["fftr_abs_percentile_20"]))))))))) * 2.0)) +

            0.040000*np.tanh(((((((((data["classic_sta_lta5_mean"]) * (((data["fftr_percentile_30"]) + (((data["fftr_num_peaks_20"]) * 2.0)))))) + (data["ffti_percentile_roll_std_40"]))) + (data["range_0_1000"]))) * 2.0)) +

            0.040000*np.tanh(((((data["ffti_abs_percentile_1"]) + (((data["fftr_hmean"]) + (((((data["fftr_num_peaks_50"]) * (data["fftr_abs_max_roll_mean_100"]))) - (data["percentile_50"]))))))) - (((data["fftr_abs_max_roll_mean_10000"]) + (data["ffti_mean_change_rate"]))))) +

            0.040000*np.tanh(((((((data["fftr_time_rev_asym_stat_10"]) * (data["fftr_range_-4000_-3000"]))) + (((((data["ffti_percentile_roll_mean_50"]) * (data["exp_Moving_average_50000_std"]))) + (data["fftr_percentile_roll_std_40"]))))) + (((data["min_roll_mean_500"]) * (data["ffti_classic_sta_lta2_mean"]))))) +

            0.040000*np.tanh(((data["abs_percentile_70"]) + (((data["fftr_exp_Moving_average_30000_mean"]) - (((((((data["fftr_exp_Moving_average_30000_mean"]) * (((((data["percentile_roll_std_20"]) * (data["percentile_90"]))) * 2.0)))) * 2.0)) * 2.0)))))) +

            0.040000*np.tanh(((data["classic_sta_lta3_mean"]) - (((data["min_first_10000"]) - (((data["ffti_autocorrelation_1000"]) * (((((data["ffti_max_roll_std_50"]) - (((((data["ffti_ave_roll_mean_500"]) * 2.0)) * 2.0)))) + (data["fftr_min_roll_std_500"]))))))))) +

            0.040000*np.tanh(((((((data["min_last_50000"]) - (data["ffti_kurt"]))) + (data["fftr_percentile_roll_std_50"]))) - (((((data["ffti_av_change_abs_roll_std_50"]) + (data["ffti_min_roll_std_50"]))) + (((data["ffti_percentile_roll_std_80"]) - (data["ffti_min_roll_std_10000"]))))))) +

            0.040000*np.tanh(((data["percentile_roll_std_60"]) - (((((data["fftr_abs_max_roll_std_50"]) + (((data["autocorrelation_100"]) + (((((data["num_peaks_10"]) * (data["num_peaks_10"]))) + (data["fftr_abs_percentile_30"]))))))) * (data["min_roll_std_1000"]))))) +

            0.040000*np.tanh(((((data["fftr_time_rev_asym_stat_10"]) * ((-1.0*((data["ffti_autocorrelation_50"])))))) + (((data["fftr_min_roll_std_500"]) + (((data["fftr_percentile_roll_mean_5"]) * (((((data["percentile_roll_std_75"]) + (data["percentile_75"]))) * 2.0)))))))) +

            0.040000*np.tanh(((((data["fftr_range_minf_m4000"]) * 2.0)) * ((-1.0*((((data["ffti_ave10"]) + (((((data["fftr_exp_Moving_average_3000_mean"]) * (((data["fftr_ave_roll_mean_10"]) + (data["fftr_abs_percentile_1"]))))) + (data["ffti_ave_roll_mean_10000"])))))))))) +

            0.040000*np.tanh(((((data["ffti_percentile_roll_std_5"]) * (data["autocorrelation_10000"]))) - (((data["ffti_abs_percentile_1"]) * (((((((data["classic_sta_lta5_mean"]) * (data["fftr_Hann_window_mean_1500"]))) + (data["ffti_av_change_abs_roll_std_10"]))) - (data["fftr_percentile_roll_mean_5"]))))))) +

            0.040000*np.tanh((((((((-1.0*((data["ffti_percentile_roll_std_60"])))) * 2.0)) * 2.0)) * (((((((data["exp_Moving_average_50000_std"]) - (((data["fftr_min_first_1000"]) * 2.0)))) + (data["fftr_exp_Moving_average_30000_mean"]))) + (data["fftr_exp_Moving_average_30000_mean"]))))) +

            0.040000*np.tanh(((data["fftr_std_last_10000"]) + (((data["ffti_min_first_1000"]) + (((data["mean_last_10000"]) * (((((data["ffti_autocorrelation_5000"]) + (((data["max_to_min_diff"]) + (data["max_to_min_diff"]))))) + (data["ffti_autocorrelation_5000"]))))))))) +

            0.040000*np.tanh(((((data["autocorrelation_500"]) * (data["ffti_Hann_window_mean_150"]))) - ((((((((data["autocorrelation_100"]) + (data["percentile_roll_mean_40"]))/2.0)) + (data["av_change_abs_roll_std_500"]))) - (((data["av_change_rate_roll_mean_10000"]) * (data["ffti_Hann_window_mean_50"]))))))) +

            0.039937*np.tanh(((((((((((data["mean_change_rate_first_1000"]) - (data["fftr_percentile_roll_mean_25"]))) - (data["ffti_abs_max_roll_std_50"]))) - (data["ffti_percentile_roll_std_75"]))) - ((((data["fftr_percentile_roll_mean_25"]) + (data["fftr_time_rev_asym_stat_5"]))/2.0)))) + (data["skew"]))) +

            0.040000*np.tanh(((((((((data["ffti_kurt"]) - (data["ffti_percentile_roll_std_10"]))) - (data["fftr_percentile_roll_std_40"]))) - (((data["fftr_percentile_roll_std_40"]) * 2.0)))) * (((data["fftr_percentile_roll_mean_95"]) + (((data["fftr_mean_last_50000"]) * 2.0)))))) +

            0.039969*np.tanh(((((data["av_change_abs_roll_mean_100"]) * (((((data["autocorrelation_1000"]) + (((data["fftr_percentile_roll_mean_99"]) + (data["ffti_max_roll_mean_10000"]))))) + (data["std_roll_std_10"]))))) + (((data["ffti_trend"]) + (data["autocorrelation_50"]))))) +

            0.040000*np.tanh(((data["fftr_percentile_roll_mean_95"]) * (((((data["fftr_percentile_roll_mean_80"]) * 2.0)) + (((((((((((data["ffti_min_first_1000"]) * 2.0)) - (data["av_change_abs_roll_mean_1000"]))) + (data["min_roll_std_500"]))) * 2.0)) * 2.0)))))) +

            0.040000*np.tanh(((data["fftr_mean_last_1000"]) * (((((((data["mean_first_1000"]) - ((((((((((data["ffti_min_first_1000"]) + (data["med"]))/2.0)) * (data["mean_change_rate_first_50000"]))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)))) +

            0.040000*np.tanh(((((data["percentile_roll_std_10"]) * ((-1.0*((((data["max_to_min"]) + (data["autocorrelation_50"])))))))) + (((((data["ave10"]) * (data["autocorrelation_50"]))) * (((data["ffti_min_first_1000"]) * 2.0)))))) +

            0.037186*np.tanh(((((((data["ffti_max_to_min"]) + (data["av_change_abs_roll_mean_500"]))) - (((((((((data["fftr_max_roll_mean_10000"]) * (((data["range_1000_2000"]) - (data["ffti_percentile_roll_mean_95"]))))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) +

            0.040000*np.tanh((((((((data["fftr_av_change_abs_roll_std_10"]) + (data["ffti_min_last_1000"]))/2.0)) + (((((data["percentile_roll_mean_25"]) * (data["ffti_max_to_min"]))) * 2.0)))) + (((((data["ffti_av_change_abs_roll_mean_10000"]) * (data["ffti_max_to_min"]))) * 2.0)))) +

            0.040000*np.tanh((((((((((((data["max_roll_mean_10000"]) + (data["ffti_av_change_abs_roll_mean_10000"]))/2.0)) * 2.0)) * 2.0)) * 2.0)) * (((((data["fftr_abs_percentile_5"]) - (((data["ffti_max_first_1000"]) * 2.0)))) - (data["fftr_spkt_welch_density_1"]))))) +

            0.039969*np.tanh(((((data["ffti_min_last_1000"]) - ((((data["min_first_10000"]) + (((data["fftr_percentile_roll_std_10"]) * 2.0)))/2.0)))) - (((data["ffti_max_roll_mean_100"]) * (((((((data["ffti_av_change_abs_roll_std_1000"]) * 2.0)) * 2.0)) * 2.0)))))) +

            0.040000*np.tanh(((data["ffti_av_change_abs_roll_std_10"]) * (((data["min_last_1000"]) + (((((((data["fftr_exp_Moving_average_50000_mean"]) + (data["ffti_range_2000_3000"]))) - (((data["abs_percentile_20"]) - (data["ffti_av_change_abs_roll_mean_1000"]))))) - (data["ffti_spkt_welch_density_5"]))))))) +

            0.040000*np.tanh(((((data["ffti_av_change_abs_roll_mean_50"]) * (((data["ffti_av_change_abs_roll_std_50"]) - (data["ffti_min_roll_std_50"]))))) + (((((data["ffti_min_roll_std_50"]) * (((data["ffti_autocorrelation_10"]) * 2.0)))) + (data["min_first_1000"]))))) +

            0.040000*np.tanh(((data["fftr_percentile_roll_std_40"]) - (((data["fftr_percentile_roll_std_40"]) * (((data["ffti_mean_change_abs"]) - (((((((data["fftr_range_3000_4000"]) - (data["spkt_welch_density_100"]))) - (data["spkt_welch_density_100"]))) - (data["ffti_mean_change_abs"]))))))))) +

            0.040000*np.tanh(((((((((((((data["mean_change_rate_last_1000"]) + (data["max_last_10000"]))/2.0)) + (data["ffti_av_change_abs_roll_mean_1000"]))/2.0)) - (((data["ffti_percentile_80"]) * (data["ffti_abs_percentile_1"]))))) - (data["max_last_10000"]))) - (data["ffti_percentile_roll_mean_50"]))) +

            0.040000*np.tanh(((((((((data["ffti_min_first_1000"]) + (data["abs_percentile_20"]))/2.0)) + (((((data["fftr_min_roll_std_50"]) - (data["ffti_av_change_abs_roll_std_10"]))) - (data["fftr_autocorrelation_10"]))))) + (data["ffti_min_first_1000"]))/2.0)) +

            0.040000*np.tanh(((((((data["ffti_percentile_roll_mean_99"]) + (((data["ffti_hmean"]) * ((-1.0*((((data["percentile_75"]) + (((data["av_change_rate_roll_std_10000"]) + (data["fftr_time_rev_asym_stat_50"])))))))))))) + (data["av_change_rate_roll_std_10000"]))) * 2.0)) +

            0.040000*np.tanh((-1.0*(((((data["ffti_percentile_roll_mean_75"]) + (((((data["ffti_min_roll_std_500"]) - (((data["ffti_min_roll_std_500"]) - (((data["percentile_50"]) * (data["ffti_min_roll_std_500"]))))))) - ((-1.0*((data["ffti_hmean"])))))))/2.0))))) +

            0.040000*np.tanh(((data["percentile_90"]) * (((data["exp_Moving_average_300_mean"]) * (((((data["percentile_roll_mean_95"]) - (((data["percentile_roll_std_25"]) * 2.0)))) - (((data["fftr_mean_last_1000"]) * (data["percentile_90"]))))))))) +

            0.040000*np.tanh(((((data["fftr_abs_percentile_1"]) * (data["fftr_time_rev_asym_stat_5"]))) + (((data["ffti_range_-1000_0"]) * (((((((((data["ffti_percentile_roll_std_25"]) + (data["av_change_rate_roll_std_100"]))) + (data["ffti_percentile_roll_std_70"]))) * 2.0)) * 2.0)))))) +

            0.033826*np.tanh(((data["fftr_av_change_abs_roll_std_10"]) + (((((((((data["ffti_percentile_roll_std_40"]) - (data["abs_percentile_20"]))) - (data["fftr_spkt_welch_density_5"]))) * 2.0)) - (((data["ffti_min_roll_std_500"]) * (data["ffti_percentile_roll_std_40"]))))))) +

            0.040000*np.tanh(((data["mean_change_rate"]) - (((data["classic_sta_lta7_mean"]) - (((((data["percentile_roll_std_75"]) * (((data["ffti_kurt"]) + (data["classic_sta_lta7_mean"]))))) - ((((data["ffti_kurt"]) + (data["classic_sta_lta7_mean"]))/2.0)))))))) +

            0.040000*np.tanh(((data["min_roll_std_10"]) * (((((((data["fftr_classic_sta_lta5_mean"]) - (((data["fftr_classic_sta_lta5_mean"]) * ((-1.0*((data["fftr_classic_sta_lta5_mean"])))))))) - (data["ffti_av_change_abs_roll_std_10"]))) - (((data["fftr_mean"]) / 2.0)))))) +

            0.040000*np.tanh((((((((data["fftr_classic_sta_lta8_mean"]) + (data["autocorrelation_100"]))/2.0)) * (((((data["ffti_av_change_abs_roll_std_10000"]) + (((data["ffti_mean_change_rate"]) + (data["ffti_kstat_1"]))))) * 2.0)))) * 2.0)) +

            0.040000*np.tanh(((((data["fftr_range_2000_3000"]) * (((data["fftr_num_peaks_20"]) - (((data["ffti_abs_percentile_10"]) + (((data["ffti_abs_percentile_5"]) * (data["fftr_range_2000_3000"]))))))))) - (((data["abs_percentile_30"]) - (data["fftr_range_2000_3000"]))))) +

            0.040000*np.tanh(((data["ffti_autocorrelation_5000"]) * (((((data["max_roll_mean_10"]) * (((data["classic_sta_lta2_mean"]) * 2.0)))) + (((data["ffti_min_roll_mean_10"]) + (((data["ffti_min_roll_mean_10"]) + (data["trend"]))))))))) +

            0.039937*np.tanh((((((((data["classic_sta_lta3_mean"]) * (data["autocorrelation_50"]))) + (((data["autocorrelation_50"]) * (data["fftr_abs_max_roll_mean_100"]))))) + (((((((data["ffti_kstat_1"]) * (data["ffti_av_change_abs_roll_mean_10"]))) * 2.0)) * 2.0)))/2.0)) +

            0.039828*np.tanh(((((((((data["std_last_10000"]) + (data["fftr_percentile_roll_std_10"]))) * 2.0)) * (((((data["autocorrelation_10000"]) + (((data["percentile_95"]) + (data["ffti_mean_first_50000"]))))) + (data["percentile_95"]))))) * 2.0)) +

            0.039984*np.tanh(((((data["ffti_range_-4000_-3000"]) * (data["fftr_percentile_roll_mean_5"]))) + (((data["ffti_med"]) * (((((((data["fftr_min_roll_mean_50"]) + (((data["ffti_abs_std"]) + (data["ffti_min_roll_std_10"]))))) * 2.0)) * 2.0)))))) +

            0.039734*np.tanh((((-1.0*((data["ffti_autocorrelation_500"])))) - ((((((-1.0*((((data["ffti_autocorrelation_500"]) * (((((data["ffti_c3_100"]) - (data["ffti_av_change_abs_roll_std_500"]))) - (data["ffti_max_to_min"])))))))) * 2.0)) * 2.0)))) +

            0.040000*np.tanh(((data["abs_percentile_20"]) + (((((data["ffti_autocorrelation_100"]) * ((((data["ffti_num_peaks_10"]) + (((((data["ffti_percentile_roll_std_30"]) * 2.0)) * (((-2.0) - (data["exp_Moving_average_50000_mean"]))))))/2.0)))) * 2.0)))) +

            0.039984*np.tanh(((data["spkt_welch_density_100"]) * (((data["fftr_min_roll_std_500"]) + (((((((data["spkt_welch_density_50"]) - (((data["exp_Moving_average_3000_std"]) * (data["ffti_exp_Moving_average_3000_mean"]))))) - (data["fftr_autocorrelation_1000"]))) + (data["ffti_exp_Moving_average_3000_mean"]))))))) +

            0.039984*np.tanh(((data["ffti_range_3000_4000"]) * (((((data["fftr_percentile_roll_mean_80"]) + ((((((data["fftr_percentile_roll_mean_80"]) + (data["fftr_percentile_roll_mean_80"]))/2.0)) * (data["fftr_min_roll_mean_1000"]))))) + (data["ffti_percentile_roll_std_5"]))))) +

            0.039984*np.tanh(((data["ffti_min_roll_mean_1000"]) * (((data["ffti_av_change_abs_roll_std_10"]) + (((((((data["fftr_iqr"]) * (data["fftr_percentile_roll_mean_5"]))) + ((-1.0*((data["min_first_10000"])))))) + ((-1.0*((data["min_roll_std_50"])))))))))) +

            0.040000*np.tanh(((data["ffti_percentile_roll_mean_50"]) * (((((data["ffti_autocorrelation_100"]) + (((data["spkt_welch_density_100"]) + (data["abs_trend"]))))) + (((((data["min_first_50000"]) + (data["spkt_welch_density_100"]))) - (data["ffti_percentile_roll_std_30"]))))))) +

            0.040000*np.tanh(((((((data["classic_sta_lta3_mean"]) * (data["fftr_kstat_1"]))) + (((data["max_first_10000"]) - (data["ffti_av_change_abs_roll_std_500"]))))) + (((data["fftr_sum"]) * (((data["ffti_autocorrelation_50"]) + (data["fftr_hmean"]))))))) +

            0.040000*np.tanh(((((data["ffti_abs_percentile_5"]) * ((-1.0*(((((data["fftr_percentile_roll_std_95"]) + (((data["fftr_range_-3000_-2000"]) * 2.0)))/2.0))))))) - (((data["ffti_max_roll_mean_10"]) + (((data["mean_change_rate_first_10000"]) * (data["min_roll_std_1000"]))))))) +

            0.040000*np.tanh(((((data["autocorrelation_50"]) * (((data["mean_change_rate_last_1000"]) / 2.0)))) + (((data["fftr_num_peaks_10"]) * ((-1.0*((((data["autocorrelation_50"]) * (((data["exp_Moving_average_3000_std"]) + (data["percentile_roll_mean_70"])))))))))))) +

            0.040000*np.tanh(((data["fftr_percentile_roll_std_60"]) + (((((data["fftr_time_rev_asym_stat_100"]) * (((((data["hmean"]) * (data["fftr_percentile_50"]))) - (((data["fftr_mean_first_50000"]) - (data["fftr_percentile_50"]))))))) * 2.0)))) +

            0.039984*np.tanh(((((data["ffti_av_change_abs_roll_std_500"]) * (((((data["percentile_75"]) - (((data["ffti_min_first_1000"]) - ((((((((data["ffti_av_change_abs_roll_std_50"]) + (data["ffti_sum"]))/2.0)) * 2.0)) * 2.0)))))) * 2.0)))) * 2.0)) +

            0.039969*np.tanh(((data["spkt_welch_density_50"]) * (((((data["fftr_std_roll_mean_50"]) - (data["spkt_welch_density_50"]))) + (((((data["fftr_std_roll_mean_50"]) - (data["fftr_min_roll_std_50"]))) + (((data["fftr_std_roll_mean_50"]) - (data["ffti_trend"]))))))))) +

            0.040000*np.tanh((((((((((data["ffti_max_to_min"]) + (data["percentile_60"]))/2.0)) + (data["max_to_min"]))/2.0)) + (((((data["percentile_75"]) * (((data["ffti_percentile_roll_mean_40"]) * (data["percentile_roll_std_40"]))))) * 2.0)))/2.0)) +

            0.039984*np.tanh(((data["fftr_percentile_roll_mean_90"]) * (((((((((((((data["ffti_percentile_roll_mean_60"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * (((data["ffti_Hann_window_mean_1500"]) - (((data["fftr_max_last_1000"]) * 2.0)))))))) +

            0.040000*np.tanh(((((data["min_first_1000"]) * (((data["ffti_av_change_abs_roll_std_10000"]) * (((data["percentile_roll_std_50"]) - (((data["ffti_range_minf_m4000"]) * ((((data["ffti_Hann_window_mean_150"]) + (data["ffti_range_minf_m4000"]))/2.0)))))))))) * 2.0)) +

            0.040000*np.tanh(((((data["ffti_min_roll_std_10"]) * (((((((((((data["ffti_min_roll_std_10"]) * (data["ffti_min_roll_std_10"]))) + (data["av_change_abs_roll_mean_500"]))) / 2.0)) - (data["ffti_num_peaks_50"]))) + (data["std_last_50000"]))))) * 2.0)) +

            0.040000*np.tanh(((data["ffti_num_peaks_50"]) * (((((((data["percentile_roll_mean_50"]) + (((data["fftr_time_rev_asym_stat_50"]) * 2.0)))) + (data["fftr_time_rev_asym_stat_50"]))) + (((data["ffti_min_roll_std_10000"]) * (((data["fftr_max_roll_std_10"]) * 2.0)))))))) +

            0.039969*np.tanh(((((data["max_to_min"]) * (((data["ffti_c3_1000"]) + (data["classic_sta_lta4_mean"]))))) + ((((((data["classic_sta_lta4_mean"]) + (data["ffti_autocorrelation_5"]))/2.0)) + (((data["ffti_autocorrelation_5"]) * (data["fftr_min_roll_mean_500"]))))))) +

            0.040000*np.tanh(((data["fftr_spkt_welch_density_10"]) * (((((data["ffti_mean_first_50000"]) * 2.0)) + (((data["fftr_min_last_1000"]) + (((((data["fftr_min_last_1000"]) + (((data["ffti_ave_roll_mean_50"]) - (data["ffti_min_roll_std_500"]))))) * 2.0)))))))) +

            0.039984*np.tanh(((((((((data["fftr_max_roll_std_50"]) * (data["fftr_percentile_roll_mean_75"]))) * 2.0)) + (data["ffti_percentile_roll_mean_70"]))) + (((((((((data["ffti_mean_change_rate_last_10000"]) * 2.0)) * 2.0)) * 2.0)) * (data["fftr_percentile_roll_mean_75"]))))) +

            0.040000*np.tanh(((((data["fftr_spkt_welch_density_1"]) + (data["fftr_spkt_welch_density_1"]))) * (((data["fftr_num_peaks_20"]) + (((((data["fftr_spkt_welch_density_1"]) - (data["autocorrelation_500"]))) + (((data["fftr_min_roll_mean_10"]) + (data["autocorrelation_10000"]))))))))) +

            0.040000*np.tanh(((((((data["av_change_rate_roll_mean_50"]) * (((data["ffti_percentile_roll_mean_10"]) + (((((data["fftr_range_1000_2000"]) + (data["fftr_percentile_roll_mean_40"]))) - (((data["ffti_percentile_roll_mean_75"]) - (data["av_change_abs_roll_mean_1000"]))))))))) * 2.0)) * 2.0)) +

            0.038156*np.tanh(((data["fftr_min_last_1000"]) * ((((((((((-1.0*(((((data["ffti_exp_Moving_average_30000_mean"]) + (((data["fftr_autocorrelation_1000"]) + (data["av_change_rate_roll_mean_10000"]))))/2.0))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) +

            0.040000*np.tanh((((((((data["mean_change_rate_last_10000"]) + (((data["ffti_percentile_roll_mean_50"]) + (((data["fftr_num_peaks_100"]) + (data["classic_sta_lta5_mean"]))))))/2.0)) / 2.0)) - ((-1.0*((((data["ffti_mean_last_10000"]) * (data["min_roll_std_10"])))))))) +

            0.039953*np.tanh(((data["ffti_autocorrelation_5"]) * (((((data["ffti_percentile_roll_std_50"]) + (((data["fftr_percentile_40"]) + (data["fftr_min_first_1000"]))))) + ((((((data["fftr_min_first_1000"]) + (data["fftr_percentile_40"]))/2.0)) + (data["ffti_percentile_roll_std_50"]))))))) +

            0.040000*np.tanh(((((((((-1.0*((data["fftr_mean_change_rate_first_10000"])))) * 2.0)) * 2.0)) + (((data["max_first_50000"]) + ((((data["skew"]) + (((data["classic_sta_lta3_mean"]) + (data["ffti_classic_sta_lta5_mean"]))))/2.0)))))/2.0)) +

            0.040000*np.tanh(((data["percentile_roll_mean_1"]) * (((data["classic_sta_lta6_mean"]) + ((-1.0*(((((data["min_roll_std_1000"]) + ((-1.0*((((data["av_change_abs_roll_mean_10000"]) * (((data["classic_sta_lta6_mean"]) + (data["percentile_60"])))))))))/2.0))))))))) +

            0.040000*np.tanh(((data["percentile_roll_mean_90"]) * ((-1.0*(((((((data["fftr_abs_percentile_1"]) + (data["ffti_percentile_roll_std_60"]))) + (((data["fftr_time_rev_asym_stat_100"]) + ((((-1.0*((data["fftr_range_-3000_-2000"])))) + (data["fftr_autocorrelation_500"]))))))/2.0))))))) +

            0.039984*np.tanh(((((((data["exp_Moving_average_3000_std"]) * (((data["fftr_min_roll_std_500"]) - (data["ffti_min_roll_std_50"]))))) * 2.0)) + (((((data["ffti_abs_max_roll_mean_500"]) + (data["fftr_av_change_abs_roll_mean_10000"]))) * (data["ffti_abs_max_roll_mean_500"]))))) +

            0.038671*np.tanh(((((((data["classic_sta_lta4_mean"]) * (((data["fftr_hmean"]) - (data["fftr_min_roll_mean_10"]))))) + (data["fftr_min_roll_std_500"]))) + (((((data["fftr_min_roll_std_500"]) + (data["fftr_hmean"]))) * (data["ffti_min_roll_mean_1000"]))))) +

            0.040000*np.tanh((-1.0*((((data["fftr_percentile_roll_mean_30"]) * (((data["ffti_num_peaks_100"]) + (((((data["max_first_10000"]) + (data["ffti_num_peaks_100"]))) + (((data["abs_percentile_20"]) + (((data["ffti_min_roll_std_10"]) / 2.0))))))))))))) +

            0.040000*np.tanh(((data["fftr_time_rev_asym_stat_10"]) * (((((data["av_change_abs_roll_std_10"]) + (((((data["classic_sta_lta3_mean"]) + (data["fftr_num_peaks_10"]))) + (data["fftr_time_rev_asym_stat_5"]))))) + (((data["classic_sta_lta3_mean"]) * (data["ffti_min_roll_mean_50"]))))))) +

            0.039969*np.tanh(((((((data["percentile_roll_mean_75"]) * (data["percentile_60"]))) * (((data["classic_sta_lta2_mean"]) + (data["ffti_min_roll_mean_500"]))))) + (((data["percentile_60"]) * (((data["ffti_autocorrelation_1000"]) - (data["ffti_percentile_50"]))))))) +

            0.039922*np.tanh(((data["percentile_60"]) * (((data["fftr_abs_max_roll_std_100"]) + (((data["classic_sta_lta2_mean"]) + (((((data["classic_sta_lta2_mean"]) + (((data["ffti_max_roll_std_10000"]) - (data["fftr_std_roll_mean_10000"]))))) - (data["fftr_std_roll_mean_10000"]))))))))) +

            0.039984*np.tanh(((((((data["fftr_min_roll_mean_10000"]) * ((((-1.0*((data["autocorrelation_100"])))) + (((((data["fftr_min_roll_mean_100"]) / 2.0)) + ((-1.0*((data["fftr_min_roll_mean_10000"])))))))))) * 2.0)) + (data["av_change_rate_roll_std_500"]))) +

            0.039937*np.tanh((-1.0*((((data["percentile_60"]) * ((((((data["num_peaks_100"]) - ((((-1.0*((data["ffti_autocorrelation_10000"])))) + (data["ffti_abs_percentile_50"]))))) + (((data["num_peaks_100"]) - (data["classic_sta_lta4_mean"]))))/2.0))))))) +

            0.039984*np.tanh(((((((data["av_change_rate_roll_std_10"]) * 2.0)) * (data["max_to_min"]))) - (((data["autocorrelation_5000"]) * (((((data["fftr_percentile_25"]) - (data["max_to_min"]))) - (data["fftr_abs_max_roll_std_100"]))))))) +

            0.039984*np.tanh(((((((((data["fftr_autocorrelation_500"]) * (data["fftr_autocorrelation_5000"]))) + (((data["ffti_min_roll_std_10000"]) * (data["percentile_75"]))))) + (data["std_last_50000"]))) + (((data["ffti_med"]) * (data["av_change_rate_roll_std_500"]))))) +

            0.040000*np.tanh(((data["percentile_50"]) * (((((((data["ffti_max_to_min"]) + (data["fftr_range_2000_3000"]))/2.0)) + ((((data["fftr_range_2000_3000"]) + (((data["ffti_max_to_min"]) * (((data["ffti_max_to_min"]) * (data["percentile_50"]))))))/2.0)))/2.0)))) +

            0.039984*np.tanh(((data["fftr_percentile_roll_std_20"]) * (((((((data["ffti_autocorrelation_500"]) - (data["ffti_percentile_roll_mean_60"]))) - (((((data["ffti_autocorrelation_500"]) - (data["ffti_percentile_roll_mean_60"]))) * (data["fftr_ave_roll_std_100"]))))) - (data["fftr_percentile_roll_mean_50"]))))) +

            0.040000*np.tanh((-1.0*((((((data["fftr_mean_change_rate_first_10000"]) - (((data["ffti_num_peaks_100"]) * (((((((data["ffti_percentile_roll_std_30"]) + (((data["fftr_max_roll_mean_1000"]) + (data["ffti_autocorrelation_50"]))))/2.0)) + (data["fftr_max_roll_mean_1000"]))/2.0)))))) * 2.0))))) +

            0.039984*np.tanh(((data["std_first_10000"]) * (((((data["fftr_min_roll_std_100"]) - (((data["min_roll_std_500"]) * (data["ffti_classic_sta_lta6_mean"]))))) + (((data["fftr_min_roll_std_1000"]) - ((-1.0*((data["fftr_mean_last_50000"])))))))))) +

            0.040000*np.tanh(((((data["fftr_moment_4"]) + (data["ffti_min_roll_std_10"]))) * (((((data["ffti_exp_Moving_average_50000_std"]) + (data["std_last_50000"]))) + (((data["ffti_min_roll_std_500"]) + (data["mean_change_rate_first_10000"]))))))) +

            0.039984*np.tanh((((((data["av_change_abs_roll_std_1000"]) * (((data["max_to_min"]) - (((data["fftr_Hann_window_mean_1500"]) - (((data["av_change_abs_roll_std_1000"]) * (data["ffti_percentile_roll_std_40"]))))))))) + (((data["ffti_percentile_roll_std_40"]) - (data["ffti_autocorrelation_10000"]))))/2.0)) +

            0.039984*np.tanh((((((((data["ffti_percentile_roll_std_70"]) * (((data["ffti_mean_last_1000"]) * 2.0)))) + ((((((data["fftr_min_first_1000"]) * (data["fftr_spkt_welch_density_10"]))) + (((data["ffti_min_roll_std_1000"]) * (data["ffti_percentile_roll_mean_25"]))))/2.0)))/2.0)) * 2.0)) +

            0.039984*np.tanh(((((data["ffti_std_first_50000"]) * (((data["fftr_kstat_1"]) * (((((((data["ffti_min_roll_std_10"]) - (data["abs_percentile_80"]))) * 2.0)) + (((data["fftr_mean_last_1000"]) - (data["fftr_std_roll_mean_10000"]))))))))) * 2.0)) +

            0.039891*np.tanh(((((((data["max_last_10000"]) + (data["skew"]))) * (data["ffti_min_roll_std_10"]))) + (((data["fftr_time_rev_asym_stat_100"]) * (((((data["ffti_percentile_roll_std_30"]) + (data["max_last_50000"]))) + (data["max_last_10000"]))))))) +

            0.039953*np.tanh(((data["fftr_abs_percentile_1"]) * (((((((data["ffti_range_2000_3000"]) * (((data["ffti_range_2000_3000"]) * (data["fftr_num_peaks_100"]))))) - (data["ffti_percentile_roll_std_30"]))) - (data["ffti_percentile_roll_std_30"]))))) +

            0.040000*np.tanh((((((((data["max_last_1000"]) + (data["ffti_percentile_roll_std_10"]))) + (data["ffti_autocorrelation_100"]))/2.0)) * ((((((((data["ffti_autocorrelation_100"]) + (data["av_change_abs_roll_std_10"]))/2.0)) + (data["percentile_70"]))) + (data["percentile_70"]))))) +

            0.039984*np.tanh((((((((data["av_change_abs_roll_mean_50"]) * ((((data["ffti_percentile_roll_mean_70"]) + ((((data["av_change_abs_roll_std_10"]) + (data["fftr_min_roll_std_100"]))/2.0)))/2.0)))) * 2.0)) + (((data["fftr_abs_percentile_1"]) * (((data["ffti_percentile_roll_mean_70"]) * 2.0)))))/2.0)) +

            0.039969*np.tanh(((data["fftr_c3_10000"]) - (((((data["ffti_percentile_roll_std_10"]) + (data["ffti_percentile_roll_std_10"]))) * (((data["fftr_sum"]) - (((data["fftr_percentile_roll_mean_70"]) - (((data["abs_percentile_20"]) + (data["fftr_Hann_window_mean_50"]))))))))))) +

            0.039500*np.tanh(((data["ffti_range_-4000_-3000"]) + (((data["min_roll_std_500"]) + (((((data["min_roll_std_500"]) + (data["ffti_abs_percentile_70"]))) * (((data["min_roll_std_500"]) * (((data["max_roll_mean_100"]) - (data["min_roll_std_500"]))))))))))) +

            0.040000*np.tanh(((((data["fftr_min_roll_std_500"]) - (data["ffti_av_change_abs_roll_std_50"]))) * (((data["ffti_av_change_abs_roll_std_50"]) + (((data["ffti_percentile_25"]) + (((data["percentile_roll_mean_5"]) + ((-1.0*((data["ffti_abs_percentile_5"])))))))))))) +

            0.040000*np.tanh(((((data["mean_last_10000"]) + (data["min_roll_std_10"]))) * ((((data["mean_change_abs"]) + ((((data["ffti_autocorrelation_5000"]) + (((data["min_roll_std_10"]) * ((((data["mean_change_rate_last_10000"]) + (data["ffti_av_change_abs_roll_mean_10000"]))/2.0)))))/2.0)))/2.0)))) +

            0.039218*np.tanh((-1.0*((((data["num_peaks_50"]) * ((((((data["fftr_ave_roll_mean_500"]) + (data["mean_change_rate_last_1000"]))) + (data["ffti_num_peaks_10"]))/2.0))))))) +

            0.040000*np.tanh(((data["fftr_autocorrelation_50"]) * ((((data["ffti_min_last_1000"]) + (((((data["max_first_10000"]) + (((((data["ffti_min_last_1000"]) - (data["mean_change_rate_first_10000"]))) + (data["ffti_num_peaks_50"]))))) - (data["ffti_autocorrelation_10"]))))/2.0)))) +

            0.039984*np.tanh(((((data["av_change_rate_roll_mean_500"]) * (((data["ffti_num_peaks_50"]) - (((data["ffti_mean_last_10000"]) - (((data["fftr_time_rev_asym_stat_1"]) + (data["fftr_time_rev_asym_stat_1"]))))))))) + (((data["autocorrelation_5000"]) * (data["ffti_classic_sta_lta6_mean"]))))) +

            0.039969*np.tanh(((((data["ffti_percentile_roll_mean_60"]) * (((data["fftr_percentile_roll_mean_5"]) + ((((((data["ffti_av_change_abs_roll_std_10000"]) * (data["ffti_av_change_abs_roll_mean_500"]))) + (data["av_change_abs_roll_std_10"]))/2.0)))))) + (((data["ffti_abs_percentile_1"]) * (data["ffti_av_change_abs_roll_std_10000"]))))) +

            0.034639*np.tanh(((data["av_change_abs_roll_std_10"]) * ((-1.0*((((((((((data["fftr_spkt_welch_density_5"]) - (data["skew"]))) + (((data["ffti_av_change_abs_roll_std_10"]) * 2.0)))) + (((data["ffti_av_change_abs_roll_std_10"]) * 2.0)))) * 2.0))))))) +

            0.038921*np.tanh(((data["av_change_rate_roll_std_10000"]) + (((data["abs_max_roll_mean_1000"]) + (((data["fftr_c3_10000"]) + (((((data["abs_max_roll_mean_1000"]) + (data["av_change_rate_roll_std_10000"]))) * (((data["abs_percentile_5"]) + (data["min_roll_mean_10000"]))))))))))) +

            0.040000*np.tanh(((data["percentile_70"]) * (((((((data["av_change_abs_roll_mean_10"]) + (((data["abs_percentile_10"]) - (data["mean_change_rate_last_1000"]))))/2.0)) + ((((data["fftr_av_change_abs_roll_std_10000"]) + (data["fftr_autocorrelation_5"]))/2.0)))/2.0)))) +

            0.037921*np.tanh(((data["fftr_time_rev_asym_stat_50"]) * ((((((data["fftr_std_roll_mean_10"]) + (data["abs_percentile_40"]))/2.0)) + (((data["ffti_percentile_roll_mean_5"]) + (((data["min_roll_std_50"]) + (data["av_change_abs_roll_mean_10"]))))))))) +

            0.040000*np.tanh(((((data["ffti_abs_percentile_5"]) * (data["ffti_skew"]))) + (((((data["fftr_min_first_1000"]) - (data["ffti_skew"]))) * ((((data["fftr_autocorrelation_10000"]) + ((((data["exp_Moving_average_50000_std"]) + (data["ffti_spkt_welch_density_10"]))/2.0)))/2.0)))))) +

            0.039984*np.tanh(((((((data["ffti_av_change_abs_roll_std_10000"]) + (data["percentile_roll_std_25"]))) * (((data["fftr_percentile_roll_std_50"]) - ((((data["ffti_ave10"]) + (((data["ffti_max_roll_mean_100"]) * (data["percentile_roll_std_20"]))))/2.0)))))) - (data["c3_100"]))) +

            0.040000*np.tanh((((data["autocorrelation_50"]) + (((data["ffti_mean_change_rate_last_10000"]) * (((data["abs_percentile_40"]) - (((((data["autocorrelation_50"]) + (data["autocorrelation_100"]))) + (data["autocorrelation_1000"]))))))))/2.0)) +

            0.036561*np.tanh(((((((((((data["ffti_med"]) + (data["fftr_time_rev_asym_stat_100"]))) + (data["av_change_rate_roll_mean_50"]))) * (((data["ffti_num_peaks_50"]) + (((data["count_big"]) + (data["ffti_percentile_roll_mean_75"]))))))) * 2.0)) * 2.0)) +

            0.037796*np.tanh(((((data["ffti_abs_max_roll_std_100"]) - (data["ffti_min_roll_std_50"]))) * (((data["ffti_min_roll_std_50"]) + ((((((data["ffti_num_peaks_50"]) + (data["fftr_kstat_3"]))/2.0)) + (((data["ffti_autocorrelation_100"]) - (data["abs_percentile_10"]))))))))) +

            0.039906*np.tanh((((((((data["max_first_1000"]) * (data["max_first_1000"]))) - ((-1.0*(((0.67554849386215210))))))) + ((-1.0*(((((data["ffti_min_roll_std_50"]) + ((0.67554849386215210)))/2.0))))))/2.0)) +

            0.039984*np.tanh(((((data["fftr_percentile_roll_std_25"]) - ((((((data["fftr_percentile_roll_std_5"]) + (data["percentile_60"]))/2.0)) + ((((-1.0*((data["ffti_ave_roll_mean_10"])))) + (data["ffti_c3_5000"]))))))) * (data["fftr_percentile_roll_std_5"]))) +

            0.039969*np.tanh(((data["fftr_time_rev_asym_stat_50"]) * (((data["ffti_kstat_1"]) + (((data["ffti_kstat_1"]) + (((((data["ffti_min_roll_mean_10"]) + (((data["fftr_percentile_50"]) - (data["percentile_roll_std_50"]))))) * (data["ffti_min_roll_mean_10"]))))))))) +

            0.039953*np.tanh(((data["fftr_percentile_roll_std_40"]) * (((((((((((data["skew"]) - (data["min_roll_std_1000"]))) - (data["fftr_time_rev_asym_stat_100"]))) - (data["ffti_exp_Moving_average_30000_std"]))) - (data["ffti_mean_last_50000"]))) - (data["ffti_c3_1000"]))))) +

            0.039719*np.tanh(((((((data["autocorrelation_100"]) * (((data["ffti_av_change_abs_roll_std_500"]) * (((data["abs_max_roll_std_1000"]) + ((((((data["ffti_min_roll_std_100"]) + (data["ffti_max_roll_mean_50"]))/2.0)) * 2.0)))))))) * 2.0)) * 2.0)) +

            0.039969*np.tanh(((((data["ffti_percentile_50"]) * (((data["ffti_av_change_abs_roll_mean_10000"]) + ((((((data["ffti_sum"]) + (((data["av_change_abs_roll_mean_500"]) * (data["ffti_percentile_50"]))))/2.0)) * 2.0)))))) - (data["ffti_sum"]))) +

            0.040000*np.tanh(((data["fftr_hmean"]) * (((data["fftr_percentile_roll_std_60"]) + (((data["ffti_percentile_roll_mean_70"]) + ((((((data["ffti_classic_sta_lta6_mean"]) + (data["fftr_Hann_window_mean_150"]))) + (((data["ffti_percentile_roll_mean_75"]) * (data["fftr_Hann_window_mean_150"]))))/2.0)))))))) +

            0.040000*np.tanh(((data["fftr_Hann_window_mean_15000"]) * (((data["fftr_std_roll_mean_10000"]) + (((((((data["autocorrelation_10"]) * 2.0)) * (((data["av_change_rate_roll_std_10000"]) + ((((data["fftr_std_roll_mean_10000"]) + (data["fftr_Hann_window_mean_50"]))/2.0)))))) / 2.0)))))) +

            0.040000*np.tanh(((data["fftr_percentile_roll_std_50"]) * (((data["fftr_time_rev_asym_stat_100"]) + (((data["fftr_percentile_roll_std_50"]) + (((data["fftr_min"]) + (((((data["ffti_min_roll_std_1000"]) + ((-1.0*((data["ffti_percentile_roll_mean_20"])))))) * 2.0)))))))))) +

            0.039203*np.tanh(((((data["ffti_mean_change_rate"]) * (((data["fftr_time_rev_asym_stat_5"]) + (((data["fftr_autocorrelation_100"]) + (((data["ffti_min_roll_mean_10000"]) + (data["fftr_time_rev_asym_stat_10"]))))))))) * 2.0)) +

            0.037905*np.tanh(((data["ffti_kstat_1"]) * (((((((data["fftr_percentile_roll_std_10"]) + (data["ffti_std_roll_mean_50"]))) + (data["ffti_mean_change_rate_last_10000"]))) - (((((data["percentile_roll_mean_40"]) - (data["fftr_percentile_roll_std_10"]))) - (data["fftr_percentile_roll_std_10"]))))))) +

            0.040000*np.tanh(((data["ffti_autocorrelation_10000"]) * ((((((data["fftr_std_roll_std_100"]) + (((data["fftr_min_roll_std_10000"]) * (data["fftr_min_roll_std_10000"]))))/2.0)) - ((((data["ffti_kstat_1"]) + (((data["fftr_abs_max_roll_mean_50"]) - (data["fftr_min_roll_std_10000"]))))/2.0)))))) +

            0.040000*np.tanh((-1.0*((((data["ffti_percentile_roll_std_40"]) * (((((((((data["max_last_50000"]) * 2.0)) - (((((data["autocorrelation_1000"]) - (data["ffti_percentile_roll_std_40"]))) / 2.0)))) * 2.0)) - (data["autocorrelation_1000"])))))))) +

            0.039984*np.tanh(((((data["ffti_av_change_abs_roll_std_10000"]) * (((data["fftr_c3_5"]) + (((((data["med"]) * (((data["ffti_percentile_roll_std_5"]) + (data["fftr_mean_last_10000"]))))) + (data["fftr_mean_last_10000"]))))))) * 2.0)) +

            0.039953*np.tanh(((data["ffti_percentile_roll_std_60"]) * (((((((data["fftr_min_roll_std_500"]) - (data["fftr_abs_max_roll_mean_100"]))) - (data["ffti_kurt"]))) + (((data["fftr_exp_Moving_average_30000_std"]) * (((data["fftr_min_roll_std_500"]) - (data["fftr_abs_max_roll_mean_100"]))))))))) +

            0.039984*np.tanh(((data["ffti_percentile_roll_std_30"]) * ((-1.0*((((((data["percentile_50"]) * (data["ffti_percentile_roll_std_20"]))) + (((((data["fftr_mean_change_rate_first_10000"]) - (data["ffti_num_peaks_100"]))) - (data["fftr_percentile_roll_std_10"])))))))))) +

            0.040000*np.tanh(((data["spkt_welch_density_50"]) * (((data["abs_percentile_40"]) + (((((data["fftr_percentile_roll_std_25"]) + (data["fftr_time_rev_asym_stat_100"]))) - (((data["av_change_rate_roll_mean_10000"]) - (((data["min_roll_std_500"]) * (data["percentile_roll_mean_70"]))))))))))) +

            0.040000*np.tanh(((data["ffti_max_roll_mean_1000"]) * (((data["ffti_max_roll_mean_100"]) * ((((((-1.0*((data["fftr_percentile_roll_std_5"])))) + ((((-1.0*((data["ffti_sum"])))) / 2.0)))) + ((-1.0*((data["fftr_percentile_roll_std_10"])))))))))) +

            0.039969*np.tanh(((data["autocorrelation_1000"]) * (((data["fftr_percentile_roll_std_1"]) + (((data["fftr_min_last_1000"]) + (((((data["fftr_percentile_roll_mean_5"]) + (data["fftr_percentile_roll_mean_5"]))) + (((data["ffti_percentile_roll_mean_80"]) * (data["ffti_mean_first_50000"]))))))))))) +

            0.039969*np.tanh(((data["std_first_10000"]) * (((data["autocorrelation_10000"]) + (((((data["ffti_num_peaks_10"]) * (((data["autocorrelation_10000"]) + (((data["std_first_10000"]) + (data["autocorrelation_10000"]))))))) + (data["mean_change_rate_first_1000"]))))))) +

            0.040000*np.tanh(((((((data["ffti_ave_roll_mean_50"]) * (data["fftr_std_roll_mean_50"]))) + (((((data["ffti_iqr1"]) + (data["num_peaks_10"]))) + (data["fftr_abs_percentile_1"]))))) * (((data["ffti_ave_roll_mean_50"]) * (data["ffti_percentile_roll_std_30"]))))) +

            0.039984*np.tanh(((((data["exp_Moving_average_3000_std"]) + ((((data["ffti_max_to_min"]) + (data["classic_sta_lta1_mean"]))/2.0)))) * ((((-1.0*((data["ffti_mean_change_rate_last_10000"])))) - (data["fftr_min_roll_std_100"]))))) +

            0.040000*np.tanh(((data["std_first_50000"]) * (((((((data["fftr_min_roll_std_100"]) - (data["kstat_3"]))) - ((-1.0*((((data["spkt_welch_density_50"]) - (data["fftr_time_rev_asym_stat_100"])))))))) - (data["fftr_mean_first_10000"]))))) +

            0.039984*np.tanh((((((data["min_roll_std_10"]) * (((((((data["fftr_percentile_roll_std_1"]) - (data["ffti_percentile_roll_std_25"]))) + (data["ffti_percentile_roll_mean_60"]))) - (data["fftr_abs_percentile_1"]))))) + (((data["ffti_percentile_roll_std_20"]) * (data["ffti_percentile_roll_std_25"]))))/2.0)) +

            0.039984*np.tanh(((data["ffti_percentile_roll_std_20"]) * (((data["percentile_60"]) * (((((data["fftr_ave_roll_mean_50"]) - (data["ffti_mean_first_50000"]))) - (((data["exp_Moving_average_3000_std"]) - (((data["fftr_classic_sta_lta6_mean"]) - (data["ffti_percentile_roll_std_20"]))))))))))) +

            0.038406*np.tanh((((data["abs_percentile_30"]) + ((((((((data["ffti_av_change_abs_roll_mean_50"]) / 2.0)) + ((((data["fftr_max_roll_std_1000"]) + (data["autocorrelation_5000"]))/2.0)))/2.0)) - (((data["fftr_max_roll_std_1000"]) * (((data["exp_Moving_average_50000_std"]) * 2.0)))))))/2.0)) +

            0.040000*np.tanh(((((data["ffti_skew"]) * (data["abs_percentile_20"]))) - (((data["mean_change_rate_last_10000"]) * ((((data["spkt_welch_density_50"]) + ((((data["min_last_10000"]) + (((data["spkt_welch_density_1"]) - (data["mean_change_rate_last_10000"]))))/2.0)))/2.0)))))) +

            0.040000*np.tanh(((((((data["ffti_kurt"]) * (data["ffti_kurt"]))) + (((data["num_peaks_100"]) + (((((((data["ffti_percentile_roll_mean_75"]) * (data["ffti_hmean"]))) / 2.0)) / 2.0)))))) * (data["ffti_hmean"]))) +

            0.039828*np.tanh((-1.0*((((((data["ffti_percentile_roll_std_10"]) / 2.0)) + (((data["fftr_std_last_1000"]) + (((data["fftr_std_last_1000"]) - (((data["percentile_99"]) * (((data["fftr_abs_max_roll_std_500"]) + (data["ffti_percentile_roll_std_10"])))))))))))))) +

            0.039984*np.tanh(((data["fftr_spkt_welch_density_10"]) * (((((((((data["ffti_percentile_roll_std_40"]) - (data["ffti_std_roll_std_1000"]))) + (((data["ffti_kstat_1"]) - (data["fftr_percentile_roll_mean_5"]))))/2.0)) + (((data["ffti_percentile_roll_std_40"]) - (data["fftr_percentile_roll_std_60"]))))/2.0)))) +

            0.040000*np.tanh((((((-1.0*((data["fftr_mean_change_rate_first_10000"])))) * 2.0)) - (((data["ffti_mean_last_50000"]) * (((data["abs_percentile_10"]) + (((((-1.0*((data["ffti_percentile_roll_std_1"])))) + (data["ffti_exp_Moving_std_30000_mean"]))/2.0)))))))) +

            0.040000*np.tanh(((((data["fftr_std_roll_std_100"]) * (((data["ffti_min_roll_std_100"]) * (data["ffti_min_last_1000"]))))) + (((data["std_first_10000"]) * ((((((data["abs_percentile_20"]) + (data["abs_percentile_70"]))/2.0)) + (data["ffti_max_to_min_diff"]))))))) +

            0.031278*np.tanh(((data["fftr_c3_50"]) + (((data["ffti_min_roll_std_1000"]) * (((((data["av_change_rate_roll_mean_500"]) * 2.0)) + (((((-1.0*((data["percentile_30"])))) + (data["ffti_kurt"]))/2.0)))))))) +

            0.040000*np.tanh((((((((((((((data["ffti_autocorrelation_5"]) * (data["fftr_time_rev_asym_stat_5"]))) / 2.0)) - (data["fftr_time_rev_asym_stat_50"]))) + ((-1.0*((data["fftr_autocorrelation_10"])))))/2.0)) * (data["fftr_time_rev_asym_stat_5"]))) - (data["fftr_mean_change_rate_last_1000"]))) +

            0.040000*np.tanh((((((data["ffti_av_change_abs_roll_std_10000"]) * (data["ffti_std_roll_mean_50"]))) + (((((-1.0*((data["ffti_min_roll_std_1000"])))) + (((data["ffti_c3_1000"]) + ((((-1.0*((data["ffti_min_roll_std_1000"])))) * (data["ffti_std_roll_mean_50"]))))))/2.0)))/2.0)))
def GPII(data):

    return (5.577521 +

            0.040000*np.tanh((((((((((-1.0*((((data["fftr_range_-3000_-2000"]) + (data["min_roll_std_10000"])))))) + (((((data["fftr_abs_trend"]) + (((data["range_0_1000"]) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((-1.0*((((((data["min_roll_std_10000"]) + (((data["percentile_roll_std_10"]) + (((((((data["num_peaks_10"]) + (data["ffti_range_-4000_-3000"]))) * 2.0)) + (data["iqr"]))))))) * 2.0))))) * 2.0)) +

            0.040000*np.tanh((((((((((((((data["fftr_av_change_abs_roll_std_500"]) - (((data["percentile_roll_std_10"]) + (data["ffti_range_-3000_-2000"]))))) * 2.0)) + (((((data["range_0_1000"]) * 2.0)) * 2.0)))/2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((data["ffti_range_0_1000"]) - (data["min_roll_std_10000"]))) - (data["percentile_roll_std_30"]))) - (data["num_peaks_10"]))) - (data["fftr_range_-3000_-2000"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((data["ffti_range_-1000_0"]) - (((data["min_roll_std_10000"]) + (data["num_peaks_10"]))))) - (((data["fftr_range_-4000_-3000"]) + (data["percentile_roll_std_20"]))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((((((((-1.0*((data["ffti_range_-4000_-3000"])))) - (data["percentile_roll_std_5"]))) - (data["percentile_roll_std_20"]))) * 2.0)) - (data["ffti_range_-3000_-2000"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((-1.0*((((((data["ffti_range_3000_4000"]) + (data["min_roll_std_500"]))) + (data["iqr"])))))) - (((((data["ffti_range_3000_4000"]) + (data["percentile_roll_std_10"]))) * 2.0)))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((-1.0*((((data["min_roll_std_1000"]) - (((((((data["ffti_range_-1000_0"]) - (((data["percentile_roll_std_20"]) + (data["num_peaks_10"]))))) * 2.0)) - (data["ffti_range_-4000_-3000"])))))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((-1.0*((((data["range_-1000_0"]) + (data["iqr"])))))) - (((data["ffti_range_2000_3000"]) + (data["percentile_roll_std_5"]))))) * 2.0)) - (data["min_roll_std_1000"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((-1.0*((((((data["fftr_range_-3000_-2000"]) + (((data["iqr"]) + (((((data["percentile_roll_std_5"]) + (((data["percentile_95"]) + (data["ffti_range_3000_4000"]))))) * 2.0)))))) * 2.0))))) * 2.0)) +

            0.040000*np.tanh((((((((-1.0*(((((((((data["iqr"]) + (data["fftr_range_2000_3000"]))) + (data["ffti_range_3000_4000"]))/2.0)) + (((data["num_peaks_10"]) + (data["percentile_roll_std_10"])))))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["ffti_percentile_40"]) - ((((((data["fftr_range_-3000_-2000"]) + (data["min_roll_std_100"]))/2.0)) + (((data["percentile_roll_std_5"]) + (data["num_peaks_10"]))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((data["fftr_av_change_abs_roll_mean_1000"]) + (((data["range_0_1000"]) * 2.0)))) * 2.0)) - (data["ffti_range_-4000_-3000"]))) * 2.0)) * 2.0)) + (data["ffti_range_0_1000"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((data["fftr_trend"]) + (((data["range_0_1000"]) * 2.0)))) * 2.0)) - (data["fftr_range_-3000_-2000"]))) * 2.0)) * 2.0)) - (data["fftr_range_-4000_-3000"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["range_0_1000"]) * 2.0)) - (((((data["iqr"]) * 2.0)) + (data["min_roll_std_1000"]))))) - (data["fftr_range_-3000_-2000"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["range_0_1000"]) * 2.0)) * 2.0)) - (((data["iqr"]) + (((data["hmean"]) + (((data["min_roll_std_500"]) + (data["fftr_range_2000_3000"]))))))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["fftr_sum"]) - (((((((((data["num_peaks_10"]) + (data["fftr_range_-4000_-3000"]))) + (((data["percentile_roll_std_20"]) + (data["percentile_95"]))))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["range_0_1000"]) - (((data["num_peaks_10"]) + (((((data["percentile_roll_std_30"]) + (data["fftr_range_-3000_-2000"]))) + (data["abs_percentile_75"]))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((data["mean_change_rate_last_50000"]) - (data["percentile_75"]))) - (data["num_peaks_10"]))) - (data["percentile_roll_std_10"]))) * 2.0)) * 2.0)) * 2.0)) + (data["ffti_range_0_1000"]))) * 2.0)) +

            0.040000*np.tanh((((((((((-1.0*((((data["num_peaks_10"]) * 2.0))))) - (((((data["ffti_abs_percentile_30"]) + (((data["iqr"]) + (data["min_roll_std_10000"]))))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((-2.0) + (((((((data["ffti_range_-1000_0"]) - (((data["percentile_roll_std_20"]) * 2.0)))) * 2.0)) * 2.0)))/2.0)) - (data["min_roll_std_50"]))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["ffti_range_-1000_0"]) - (((data["min_roll_std_100"]) + (((data["num_peaks_10"]) * 2.0)))))) - (data["percentile_roll_std_20"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["fftr_av_change_abs_roll_mean_10000"]) + (((((data["range_0_1000"]) - (data["fftr_range_-4000_-3000"]))) - (data["percentile_roll_std_5"]))))) * 2.0)) * 2.0)) * 2.0)) - (data["ffti_range_-4000_-3000"]))) * 2.0)) +

            0.040000*np.tanh(((((((((data["num_peaks_10"]) - (((((((((data["num_peaks_10"]) * 2.0)) + (((data["iqr"]) + (data["min_roll_std_10000"]))))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["range_0_1000"]) - (((data["mean_first_50000"]) - ((-1.0*((((data["percentile_roll_std_10"]) + (data["fftr_range_2000_3000"])))))))))) * 2.0)) * 2.0)) - (data["min_roll_std_500"]))) * 2.0)) +

            0.040000*np.tanh((-1.0*((((((((((data["num_peaks_10"]) - (((data["mean_change_rate"]) - (((data["abs_percentile_50"]) + ((((data["ffti_range_2000_3000"]) + (data["min_roll_std_1000"]))/2.0)))))))) * 2.0)) * 2.0)) * 2.0))))) +

            0.040000*np.tanh((((((((((((-1.0*((((data["ffti_range_3000_4000"]) + (((data["abs_percentile_75"]) + (data["num_peaks_10"])))))))) * 2.0)) - (data["min_roll_std_10000"]))) * 2.0)) - (data["min_roll_std_500"]))) * 2.0)) +

            0.040000*np.tanh((((((((((((-1.0*((((((data["num_peaks_10"]) + (((data["iqr"]) - (data["mean_change_rate_last_50000"]))))) * 2.0))))) - (data["abs_percentile_80"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["ffti_range_0_1000"]) - (((((data["percentile_roll_mean_75"]) - (data["mean_change_rate"]))) * 2.0)))) * 2.0)) - (data["percentile_roll_std_20"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((-1.0*((((((((((((((data["percentile_roll_std_10"]) + (((((data["ffti_range_-4000_-3000"]) + (data["abs_percentile_80"]))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0))))) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["mean_change_rate"]) - (((data["percentile_roll_mean_60"]) - (data["mean_change_rate"]))))) * 2.0)) - (data["percentile_roll_mean_60"]))) * 2.0)) * 2.0)) - (data["min_roll_std_50"]))) * 2.0)) +

            0.040000*np.tanh(((((-2.0) - (((((((((data["num_peaks_10"]) + (data["min_roll_std_500"]))) + (data["min_roll_std_100"]))) - ((-1.0*((((data["min_roll_std_10000"]) * 2.0))))))) * 2.0)))) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["range_0_1000"]) - (data["abs_percentile_80"]))) - ((((data["percentile_roll_mean_10"]) + (data["fftr_range_-3000_-2000"]))/2.0)))) * 2.0)) * 2.0)) - (data["min_roll_std_10000"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((-1.0*((((data["fftr_range_-3000_-2000"]) + (((((data["abs_percentile_50"]) + (data["percentile_roll_std_10"]))) - (data["range_0_1000"])))))))) * 2.0)) - (data["min_roll_std_50"]))) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["ffti_percentile_40"]) - ((((-1.0*((data["mean_change_rate_last_50000"])))) + (((data["num_peaks_10"]) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["mean_change_rate"]) + (((((data["mean_change_rate"]) + ((-1.0*((((data["iqr"]) + (data["num_peaks_10"])))))))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((-1.0*((((data["percentile_roll_std_10"]) + (((((data["abs_percentile_80"]) * 2.0)) - ((-1.0*((data["ffti_range_3000_4000"]))))))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((-3.0) - (((((((((data["min_roll_std_1000"]) + (((data["min_roll_std_10000"]) + (data["num_peaks_10"]))))) * 2.0)) - (data["mean_change_rate"]))) * 2.0)))) * 2.0)) +

            0.040000*np.tanh((((((((((((((((((-1.0*((((data["abs_percentile_90"]) * 2.0))))) * 2.0)) - (data["num_peaks_10"]))) * 2.0)) * 2.0)) - (data["percentile_roll_std_5"]))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((-1.0*((((data["ffti_range_2000_3000"]) + (data["percentile_roll_std_1"])))))) - (data["percentile_95"]))) - (data["abs_percentile_50"]))) * 2.0)) - (((data["percentile_roll_std_1"]) + (data["num_peaks_100"]))))) +

            0.040000*np.tanh(((((((((((((((data["range_0_1000"]) * 2.0)) - (data["ffti_range_2000_3000"]))) * 2.0)) - (((data["min_roll_std_100"]) + (data["min_roll_std_1000"]))))) - (data["min_roll_std_500"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((-1.0*((((((((data["fftr_abs_percentile_20"]) + (((((((data["abs_percentile_80"]) + (data["iqr"]))) * 2.0)) * 2.0)))) * 2.0)) + (data["percentile_roll_std_10"])))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((-2.0) - (((((((data["ffti_range_2000_3000"]) + (((((data["num_peaks_10"]) - (data["mean_change_rate"]))) * 2.0)))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((-1.0*((((((((((((((((data["abs_percentile_80"]) * 2.0)) + (data["num_peaks_10"]))) * 2.0)) + (data["min_roll_std_50"]))) + (data["min_roll_std_500"]))) * 2.0)) * 2.0))))) +

            0.040000*np.tanh(((data["abs_percentile_40"]) - (((((((((((((data["abs_percentile_40"]) * 2.0)) + (((data["num_peaks_10"]) - (((data["mean_change_rate"]) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) +

            0.040000*np.tanh(((((((((data["percentile_10"]) - (((data["abs_percentile_50"]) + (((((((data["abs_percentile_50"]) + (data["num_peaks_10"]))) + (data["percentile_roll_std_5"]))) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["range_0_1000"]) - (((data["num_peaks_10"]) - (-1.0))))) * 2.0)) - (data["Hann_window_mean_15000"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((-1.0*((((((((((((data["percentile_95"]) + (data["abs_percentile_70"]))) + (data["percentile_roll_std_1"]))) + (((((((data["fftr_percentile_roll_std_80"]) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) * 2.0))))) +

            0.040000*np.tanh((-1.0*((((((((((data["min_roll_std_500"]) + (((((((2.0) + (((data["percentile_roll_std_20"]) * 2.0)))) + (data["num_peaks_10"]))) * 2.0)))) * 2.0)) * 2.0)) * 2.0))))) +

            0.040000*np.tanh((((-1.0*((((((((data["abs_percentile_70"]) + (((data["ffti_range_2000_3000"]) + (((data["fftr_std_first_10000"]) + (((data["abs_percentile_70"]) + (data["percentile_roll_std_10"]))))))))) * 2.0)) * 2.0))))) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((((((data["abs_percentile_20"]) - (data["range_-1000_0"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) - (data["fftr_percentile_75"]))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((-1.0*((((((((((((((data["num_peaks_100"]) + (data["percentile_30"]))) + (((((((data["percentile_95"]) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0))))) +

            0.040000*np.tanh((((((((((-1.0*((((((((data["num_peaks_10"]) - (((data["mean_change_rate"]) * 2.0)))) * 2.0)) * 2.0))))) - (data["min_roll_std_500"]))) * 2.0)) * 2.0)) - (data["percentile_80"]))) +

            0.040000*np.tanh((((((((((((-1.0*((((((((data["abs_percentile_50"]) + (data["percentile_roll_std_1"]))) * 2.0)) * 2.0))))) - (data["iqr"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((9.34147071838378906)) * (((data["classic_sta_lta6_mean"]) - (((((((data["percentile_95"]) * 2.0)) + (((((data["ffti_max_first_50000"]) * (data["ffti_max_first_50000"]))) + (data["fftr_abs_percentile_75"]))))) * 2.0)))))) +

            0.040000*np.tanh(((((((((-2.0) + (-2.0))) - (((((((data["num_peaks_10"]) + ((((data["min_roll_std_100"]) + (data["percentile_90"]))/2.0)))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["ffti_mean_last_50000"]) - (((data["min_roll_std_50"]) + (((((data["percentile_90"]) * 2.0)) * 2.0)))))) - (data["autocorrelation_10"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["percentile_20"]) - ((((((data["abs_percentile_70"]) - (data["iqr"]))) + (data["ffti_range_2000_3000"]))/2.0)))) - (data["percentile_80"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["abs_percentile_70"]) * 2.0)) - (((((data["min_roll_std_10000"]) + (data["abs_percentile_50"]))) * ((9.84468555450439453)))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["abs_percentile_50"]) + (((((((((data["abs_percentile_50"]) - (data["percentile_90"]))) + (((data["mean_change_rate"]) - (data["num_peaks_10"]))))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((data["fftr_percentile_20"]) - (((data["num_peaks_10"]) - (((data["abs_percentile_20"]) - (((data["abs_percentile_25"]) - ((-1.0*((data["gmean"])))))))))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((data["mean_change_rate_first_50000"]) - (data["num_peaks_10"]))) - (data["min_roll_std_500"]))) + (((((data["mean_change_rate"]) * 2.0)) - (((data["min_roll_std_100"]) - (data["mean_change_rate_last_10000"]))))))) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["fftr_range_minf_m4000"]) - (((((((((data["percentile_90"]) * 2.0)) * 2.0)) * 2.0)) + (data["ffti_range_-2000_-1000"]))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((data["ffti_classic_sta_lta2_mean"]) - (((((((((((((data["percentile_90"]) * 2.0)) * 2.0)) - (data["ffti_classic_sta_lta2_mean"]))) - (data["abs_percentile_20"]))) * 2.0)) * 2.0)))) * 2.0)) +

            0.040000*np.tanh(((((((((data["percentile_roll_std_10"]) - (((((((data["percentile_roll_std_25"]) - (data["percentile_20"]))) * 2.0)) * 2.0)))) * 2.0)) - (((data["min_roll_std_1000"]) - (data["mean_change_rate_first_10000"]))))) * 2.0)) +

            0.040000*np.tanh(((((((((data["percentile_20"]) - ((((data["fftr_percentile_roll_std_80"]) + (((((data["fftr_percentile_roll_std_80"]) - (data["classic_sta_lta6_mean"]))) + (((data["abs_percentile_70"]) * 2.0)))))/2.0)))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((-1.0*((((((((data["ffti_av_change_abs_roll_std_10"]) + (data["percentile_80"]))) + (((data["ffti_spkt_welch_density_5"]) - (data["percentile_20"]))))) * 2.0))))) - (data["percentile_80"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((data["ffti_trend"]) + ((((-1.0*((data["classic_sta_lta7_mean"])))) + (((((((((data["abs_percentile_20"]) - (((data["percentile_90"]) * 2.0)))) * 2.0)) * 2.0)) * 2.0)))))) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["abs_percentile_50"]) - (data["range_-1000_0"]))) * 2.0)) - (data["percentile_roll_std_1"]))) + (((data["mean_change_rate_first_50000"]) - (data["abs_percentile_70"]))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((-1.0*((((data["percentile_roll_std_5"]) + (data["ffti_spkt_welch_density_1"])))))) + (((((((((data["mean_change_rate_first_50000"]) - (data["abs_percentile_25"]))) - (data["abs_percentile_50"]))) * 2.0)) * 2.0)))) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["mean_change_rate"]) - ((-1.0*((((((data["percentile_10"]) - (data["abs_percentile_5"]))) * 2.0))))))) * 2.0)) * 2.0)) + (data["mean_change_rate_last_10000"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((((((-1.0) - (data["num_peaks_10"]))) * 2.0)) * 2.0)) * 2.0)) + (data["mean_change_rate_last_10000"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["abs_percentile_20"]) + (((((((data["abs_percentile_20"]) - (((data["ffti_percentile_25"]) + ((((4.72319602966308594)) * (data["percentile_90"]))))))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["mean_change_rate_last_10000"]) - (((data["fftr_percentile_roll_mean_60"]) + (((data["ffti_mean_first_50000"]) + (((data["num_peaks_100"]) + (data["min_roll_std_100"]))))))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["exp_Moving_average_50000_mean"]) + (data["ave_roll_mean_100"]))) * 2.0)) * (data["ave_roll_mean_100"]))) - (((((((data["percentile_90"]) + (data["ave_roll_mean_100"]))) * 2.0)) * 2.0)))) * 2.0)) +

            0.040000*np.tanh((((6.0)) * ((((((6.0)) * ((-1.0*((((data["num_peaks_10"]) - (-1.0)))))))) - ((-1.0*((((data["mean_change_rate_last_50000"]) + (data["exp_Moving_average_50000_std"])))))))))) +

            0.040000*np.tanh((-1.0*((((((((((((data["fftr_percentile_roll_mean_5"]) + (((((data["percentile_roll_std_5"]) + (((data["percentile_80"]) - (data["percentile_75"]))))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0))))) +

            0.040000*np.tanh((((((((-1.0*((((((((((((data["num_peaks_10"]) + (data["abs_percentile_25"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0))))) * 2.0)) - (3.0))) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["mean_change_rate"]) / 2.0)) + (((data["ffti_iqr"]) * (((data["percentile_roll_std_5"]) * (data["range_0_1000"]))))))) - (data["range_0_1000"]))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((data["mean_change_rate_first_1000"]) + (((((((data["mean_change_rate"]) * 2.0)) * 2.0)) - (((data["min_roll_std_50"]) + (data["min_roll_std_500"]))))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((-1.0*((((((data["mean_change_rate"]) * (((((((data["ave_roll_mean_10"]) * 2.0)) + (data["ffti_abs_percentile_10"]))) * (((((((data["ffti_range_0_1000"]) * 2.0)) * 2.0)) * 2.0)))))) * 2.0))))) +

            0.040000*np.tanh(((((((((data["ffti_percentile_40"]) - (((data["fftr_percentile_roll_mean_10"]) + (((data["min_roll_std_500"]) + (((((data["fftr_percentile_roll_mean_10"]) * 2.0)) * 2.0)))))))) - (data["ffti_percentile_roll_mean_50"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((data["ffti_mean_last_50000"]) - (((((((((data["abs_percentile_75"]) - (((data["hmean"]) - (((data["ave10"]) + (((data["ffti_percentile_roll_std_80"]) * 2.0)))))))) * 2.0)) * 2.0)) * 2.0)))) +

            0.040000*np.tanh(((((((((((((((((data["mean_change_rate"]) - (data["ffti_mean_change_rate"]))) * 2.0)) - (data["ffti_av_change_abs_roll_std_10000"]))) + (data["percentile_10"]))) + (data["classic_sta_lta4_mean"]))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((data["percentile_roll_std_70"]) - (((((data["min_roll_std_10000"]) * (((((data["min_roll_std_10000"]) * (((data["percentile_80"]) + (data["ffti_abs_percentile_50"]))))) + (data["percentile_80"]))))) * 2.0)))) * 2.0)) +

            0.040000*np.tanh(((((data["fftr_percentile_75"]) * (((((((((data["fftr_ave_roll_mean_500"]) - (((data["percentile_roll_std_10"]) * (data["percentile_roll_std_1"]))))) - (data["ffti_abs_percentile_75"]))) * 2.0)) * 2.0)))) + (data["ffti_classic_sta_lta2_mean"]))) +

            0.040000*np.tanh(((((((data["percentile_roll_std_30"]) + (((data["min_roll_std_10000"]) - (((((((data["min_roll_std_10000"]) * 2.0)) * (((data["percentile_roll_std_30"]) * 2.0)))) * (data["percentile_80"]))))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["percentile_10"]) * (((data["exp_Moving_average_50000_std"]) + (data["fftr_abs_percentile_30"]))))) * 2.0)) - (((((data["ffti_percentile_roll_std_80"]) * 2.0)) * 2.0)))) + (data["percentile_roll_std_60"]))) * 2.0)) +

            0.040000*np.tanh(((data["ffti_mean_last_50000"]) + (((data["mean_change_rate_first_1000"]) + (((data["ffti_range_minf_m4000"]) - (((((((data["ffti_range_minf_m4000"]) * (data["ffti_range_minf_m4000"]))) * (data["ffti_range_minf_m4000"]))) + (data["fftr_percentile_roll_mean_75"]))))))))) +

            0.040000*np.tanh(((((((data["min_roll_std_10000"]) * ((-1.0*((((((((data["fftr_abs_percentile_60"]) * 2.0)) * (data["min_roll_std_10000"]))) * 2.0))))))) - (((data["autocorrelation_10"]) - (data["percentile_roll_std_70"]))))) * 2.0)) +

            0.040000*np.tanh(((((((((((data["skew"]) - (data["fftr_ave_roll_mean_10"]))) - (data["fftr_percentile_roll_mean_5"]))) + (((data["ffti_max_roll_std_100"]) * (((data["fftr_gmean"]) * (data["fftr_skew"]))))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((data["num_peaks_10"]) + (((((((data["percentile_roll_std_10"]) * (((((((data["fftr_trend"]) - (data["ffti_abs_mean"]))) - (data["ffti_std_roll_std_500"]))) - (data["ffti_range_-3000_-2000"]))))) * 2.0)) * 2.0)))) +

            0.040000*np.tanh(((((data["min_last_50000"]) - (((data["ffti_autocorrelation_50"]) - (((((data["autocorrelation_50"]) - (data["ffti_max_roll_mean_50"]))) - ((((data["ffti_min_roll_std_50"]) + (data["fftr_time_rev_asym_stat_5"]))/2.0)))))))) - (data["min_roll_std_100"]))) +

            0.040000*np.tanh(((((data["percentile_roll_std_70"]) + (((data["percentile_10"]) + (((((data["percentile_10"]) * 2.0)) * (((data["percentile_roll_std_70"]) * ((((-1.0*((data["fftr_av_change_abs_roll_mean_1000"])))) * 2.0)))))))))) * 2.0)) +

            0.040000*np.tanh(((((((data["percentile_roll_std_50"]) + (data["ffti_trend"]))) - (((((((data["fftr_abs_percentile_60"]) + (((data["min_roll_std_1000"]) * (data["percentile_roll_std_50"]))))) * 2.0)) * (data["ffti_percentile_60"]))))) * 2.0)) +

            0.040000*np.tanh(((((data["percentile_60"]) * (data["percentile_20"]))) + (((data["autocorrelation_5"]) + (((data["autocorrelation_5"]) - (((((((((data["fftr_percentile_roll_std_75"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)))))))) +

            0.040000*np.tanh(((data["fftr_min_roll_std_50"]) + (((data["fftr_min_roll_std_500"]) + (((data["fftr_percentile_30"]) * (((((((data["exp_Moving_average_50000_std"]) + (data["fftr_min_roll_std_50"]))) - (data["fftr_ave_roll_mean_10000"]))) - (data["fftr_percentile_30"]))))))))) +

            0.040000*np.tanh(((((((((data["abs_percentile_80"]) + (((data["abs_percentile_80"]) * ((-1.0*((((data["percentile_roll_std_10"]) * (((data["percentile_roll_std_10"]) + (data["fftr_Hann_window_mean_1500"])))))))))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["fftr_percentile_roll_mean_5"]) - (data["percentile_40"]))) - (((((data["med"]) + (data["autocorrelation_100"]))) + (((data["num_peaks_50"]) + (data["min_roll_std_50"]))))))) - (data["num_peaks_50"]))) +

            0.040000*np.tanh(((((data["ffti_percentile_roll_mean_95"]) - (((data["percentile_roll_std_80"]) * (((((data["exp_Moving_average_30000_std"]) - (data["fftr_min_roll_mean_10000"]))) * 2.0)))))) - (((data["ffti_Hann_window_mean_50"]) * (data["percentile_40"]))))) +

            0.040000*np.tanh(((((((data["percentile_60"]) * (data["ffti_percentile_roll_mean_5"]))) - (data["ffti_percentile_roll_mean_5"]))) - (((data["ffti_abs_max_roll_mean_10"]) + (((data["min_roll_std_50"]) - (((data["percentile_60"]) * (data["percentile_20"]))))))))) +

            0.040000*np.tanh(((((data["fftr_min_roll_std_500"]) + (((((((data["percentile_roll_std_70"]) + (data["ffti_range_3000_4000"]))) * 2.0)) * (((data["fftr_mean_first_50000"]) - (((data["autocorrelation_100"]) + (data["fftr_min_roll_std_500"]))))))))) * 2.0)) +

            0.040000*np.tanh(((data["autocorrelation_50"]) - (((((((data["ffti_spkt_welch_density_5"]) + (((data["ffti_percentile_roll_std_25"]) * (((((data["fftr_percentile_70"]) + (data["fftr_abs_percentile_1"]))) + (data["ffti_range_2000_3000"]))))))) * 2.0)) * 2.0)))) +

            0.040000*np.tanh(((((((data["autocorrelation_5"]) - (data["fftr_percentile_roll_mean_20"]))) + (((data["percentile_20"]) + (((((data["ffti_range_minf_m4000"]) * ((((-1.0*((data["ffti_exp_Moving_average_50000_std"])))) * 2.0)))) * 2.0)))))) * 2.0)) +

            0.040000*np.tanh(((((((data["fftr_classic_sta_lta8_mean"]) + (((((((data["fftr_percentile_roll_mean_10"]) * 2.0)) * 2.0)) - (data["skew"]))))) * 2.0)) * (((((data["ave_roll_std_10000"]) * 2.0)) + (data["ffti_max_roll_mean_1000"]))))) +

            0.040000*np.tanh(((((((((data["ffti_percentile_roll_std_1"]) - (((data["ffti_percentile_roll_std_80"]) - (((data["ffti_range_-3000_-2000"]) * (((data["fftr_exp_Moving_average_3000_mean"]) * 2.0)))))))) - (data["ffti_percentile_roll_mean_50"]))) - (data["fftr_exp_Moving_average_3000_mean"]))) * 2.0)) +

            0.040000*np.tanh(((data["fftr_percentile_roll_std_50"]) + (((data["fftr_abs_percentile_10"]) * (((((data["fftr_num_peaks_20"]) + (((data["percentile_20"]) * (((data["percentile_50"]) + (data["fftr_ave_roll_mean_10"]))))))) + (data["percentile_20"]))))))) +

            0.040000*np.tanh(((((data["fftr_hmean"]) - (((data["ffti_percentile_roll_std_50"]) * (((((((data["ffti_abs_percentile_5"]) * 2.0)) * 2.0)) * 2.0)))))) - (((data["ffti_abs_percentile_5"]) + (data["fftr_percentile_roll_std_5"]))))) +

            0.040000*np.tanh(((((data["num_peaks_10"]) * 2.0)) + (((data["abs_percentile_70"]) + (((data["exp_Moving_average_50000_std"]) - (((data["num_peaks_10"]) * (((data["num_peaks_10"]) * (((data["num_peaks_10"]) * 2.0)))))))))))) +

            0.040000*np.tanh((((((data["mean_change_rate_first_1000"]) + (data["ffti_skew"]))/2.0)) + (((((data["ffti_min_first_1000"]) + (((data["ffti_skew"]) - (((data["fftr_autocorrelation_50"]) + (data["fftr_percentile_roll_std_25"]))))))) + (data["fftr_min_roll_std_500"]))))) +

            0.040000*np.tanh(((((((((data["mean_change_rate_first_1000"]) - (((data["fftr_spkt_welch_density_5"]) * 2.0)))) - (data["med"]))) - (((data["fftr_spkt_welch_density_5"]) + (data["ffti_av_change_abs_roll_std_50"]))))) - (data["autocorrelation_10000"]))) +

            0.040000*np.tanh(((((data["ffti_percentile_roll_mean_95"]) - (((((data["fftr_range_0_1000"]) * 2.0)) * ((-1.0*((((((data["fftr_range_-4000_-3000"]) + (data["classic_sta_lta7_mean"]))) - (data["fftr_range_0_1000"])))))))))) + (data["percentile_roll_std_60"]))) +

            0.040000*np.tanh(((((((data["av_change_rate_roll_std_50"]) * 2.0)) * (((data["fftr_range_-1000_0"]) * 2.0)))) + (((data["fftr_percentile_roll_std_40"]) + (((data["min_roll_std_10000"]) * (((data["fftr_range_-1000_0"]) * (data["fftr_mean_first_50000"]))))))))) +

            0.039984*np.tanh(((((data["av_change_abs_roll_mean_1000"]) + (((((data["ffti_max_to_min"]) + ((((data["ffti_min_last_1000"]) + ((((((data["ffti_abs_percentile_1"]) + (data["av_change_abs_roll_mean_1000"]))/2.0)) - (data["percentile_75"]))))/2.0)))) * 2.0)))) * 2.0)) +

            0.040000*np.tanh(((data["mean_change_rate"]) * (((((((data["percentile_roll_mean_99"]) + (((((data["percentile_90"]) + (data["fftr_abs_percentile_50"]))) * 2.0)))) * 2.0)) * (((data["percentile_roll_mean_99"]) - (data["ffti_mean_last_10000"]))))))) +

            0.040000*np.tanh(((((((data["fftr_num_peaks_50"]) + (((((data["max_last_50000"]) * 2.0)) * 2.0)))) + (((data["fftr_max_roll_mean_100"]) + (((data["ffti_autocorrelation_5000"]) + (data["fftr_mean_last_1000"]))))))) * (data["percentile_20"]))) +

            0.040000*np.tanh(((data["skew"]) + (((((data["ffti_min_last_1000"]) + (((data["fftr_spkt_welch_density_10"]) + (data["ffti_percentile_roll_std_40"]))))) - (((data["fftr_percentile_roll_mean_50"]) - ((((data["fftr_min_roll_std_100"]) + (data["fftr_num_peaks_50"]))/2.0)))))))) +

            0.040000*np.tanh(((((((((data["ffti_mean_last_50000"]) - (data["ffti_abs_percentile_5"]))) + (((((((((data["ffti_mean_last_50000"]) + (data["mean_change_rate_last_1000"]))) + (data["autocorrelation_500"]))/2.0)) + (data["ffti_min_roll_mean_50"]))/2.0)))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((data["autocorrelation_5"]) + ((((((((data["min_roll_std_500"]) * (((((((data["ffti_percentile_roll_mean_99"]) * 2.0)) * 2.0)) * 2.0)))) + (((data["fftr_min_last_50000"]) * 2.0)))/2.0)) + (data["classic_sta_lta6_mean"]))))) +

            0.040000*np.tanh(((((((((((data["ffti_percentile_40"]) * (((data["fftr_min_roll_std_500"]) - (((((-1.0*((data["autocorrelation_5"])))) + ((-1.0*((data["ffti_min_roll_std_100"])))))/2.0)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((data["classic_sta_lta6_mean"]) * (((((data["percentile_75"]) + (data["fftr_num_peaks_20"]))) + (((data["percentile_75"]) * (((data["classic_sta_lta6_mean"]) - (((data["fftr_mean_last_50000"]) + (data["fftr_ave_roll_std_1000"]))))))))))) +

            0.040000*np.tanh(((((((data["mean_change_rate_first_10000"]) * (data["fftr_abs_max_roll_mean_50"]))) * 2.0)) - (((data["ffti_percentile_80"]) * (((((((data["fftr_range_-3000_-2000"]) + (data["fftr_max_roll_mean_10000"]))) + (data["fftr_max_roll_mean_10000"]))) * 2.0)))))) +

            0.040000*np.tanh(((data["max_first_1000"]) * (((((data["ffti_av_change_abs_roll_std_10000"]) + (data["percentile_roll_std_5"]))) * ((-1.0*(((((-1.0*((data["ffti_av_change_abs_roll_std_10000"])))) + (((data["percentile_roll_std_40"]) + (data["percentile_roll_std_5"])))))))))))) +

            0.039984*np.tanh(((((((((data["autocorrelation_50"]) * (data["ffti_autocorrelation_10000"]))) + (((data["ffti_autocorrelation_10000"]) * (data["ffti_range_-3000_-2000"]))))) + (data["abs_percentile_40"]))) * 2.0)) +

            0.039484*np.tanh(((((((((((((((data["percentile_5"]) + ((((data["ffti_percentile_75"]) + (data["av_change_abs_roll_std_10"]))/2.0)))) + (data["ffti_percentile_roll_std_50"]))) + (data["ffti_percentile_roll_std_50"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["fftr_time_rev_asym_stat_5"]) - (((data["fftr_abs_trend"]) * (data["ffti_kstat_1"]))))) * 2.0)) * 2.0)) * 2.0)) * (((data["fftr_time_rev_asym_stat_10"]) - (data["spkt_welch_density_50"]))))) +

            0.039984*np.tanh(((((data["ffti_min_roll_std_50"]) * (((data["fftr_num_peaks_100"]) + ((((((data["fftr_av_change_abs_roll_mean_1000"]) + (data["ffti_autocorrelation_10"]))/2.0)) + ((-1.0*((((data["abs_percentile_25"]) - (data["fftr_num_peaks_100"])))))))))))) * 2.0)) +

            0.040000*np.tanh(((data["min_roll_std_500"]) + (((data["num_peaks_10"]) - (((data["min_roll_std_500"]) * (((data["num_peaks_10"]) * (((data["min_roll_std_500"]) * (((data["num_peaks_10"]) * (data["min_roll_std_500"]))))))))))))) +

            0.040000*np.tanh(((((((((((((data["fftr_range_2000_3000"]) * (data["percentile_75"]))) * (data["mean_change_rate"]))) - (((data["fftr_abs_percentile_5"]) * ((-1.0*((data["percentile_20"])))))))) * 2.0)) * 2.0)) * 2.0)) +

            0.039984*np.tanh(((data["percentile_80"]) + (((data["percentile_80"]) + (((data["percentile_roll_std_40"]) * (((data["percentile_roll_std_40"]) + (((data["autocorrelation_10000"]) - (((data["percentile_75"]) * (data["percentile_80"]))))))))))))) +

            0.040000*np.tanh(((((((((data["max_to_min_diff"]) * (data["fftr_abs_max_roll_std_10"]))) * 2.0)) + (((data["ffti_range_2000_3000"]) * (((((data["fftr_percentile_20"]) - (data["ffti_range_2000_3000"]))) + (data["fftr_time_rev_asym_stat_10"]))))))) * 2.0)) +

            0.039984*np.tanh(((((data["mean_change_rate_first_10000"]) * (((((data["ffti_min_roll_mean_500"]) + (((((((data["mean_change_rate_first_50000"]) + (data["ffti_av_change_abs_roll_std_10"]))) * 2.0)) * 2.0)))) + (data["ffti_av_change_abs_roll_std_10"]))))) - (data["fftr_max_roll_mean_100"]))) +

            0.040000*np.tanh(((data["percentile_roll_mean_95"]) * (((((data["fftr_min_roll_std_500"]) - (data["percentile_90"]))) - ((((-1.0*((data["mean_change_rate_first_50000"])))) * (((data["percentile_90"]) + (data["percentile_90"]))))))))) +

            0.040000*np.tanh(((((((((data["ffti_max_roll_std_10"]) * ((-1.0*((((data["autocorrelation_50"]) - (((data["percentile_roll_std_10"]) * ((-1.0*((data["min_roll_std_1000"]))))))))))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((data["autocorrelation_500"]) * (((((data["exp_Moving_average_30000_std"]) + (data["ffti_abs_percentile_5"]))) + (data["fftr_percentile_roll_std_60"]))))) + ((((data["fftr_percentile_roll_std_20"]) + (((data["av_change_abs_roll_mean_500"]) - (data["fftr_time_rev_asym_stat_5"]))))/2.0)))) +

            0.039969*np.tanh(((((data["fftr_min_last_1000"]) * 2.0)) * (((((((data["exp_Moving_average_3000_std"]) + (((((data["ffti_spkt_welch_density_10"]) + (data["fftr_autocorrelation_10000"]))) + (data["fftr_min_last_1000"]))))) + (data["exp_Moving_average_3000_std"]))) * 2.0)))) +

            0.040000*np.tanh(((((((((data["ffti_av_change_abs_roll_mean_500"]) * (data["ffti_autocorrelation_1000"]))) + (data["fftr_percentile_roll_std_50"]))) * 2.0)) - ((-1.0*((((((data["classic_sta_lta3_mean"]) - (data["ffti_kstat_1"]))) - (data["ffti_autocorrelation_1000"])))))))) +

            0.040000*np.tanh(((data["abs_percentile_80"]) * (((((data["autocorrelation_5000"]) - (((((((data["fftr_abs_percentile_1"]) * (data["fftr_exp_Moving_average_50000_mean"]))) * 2.0)) * 2.0)))) - ((-1.0*((((data["ffti_classic_sta_lta6_mean"]) * 2.0))))))))) +

            0.039984*np.tanh(((data["exp_Moving_average_30000_std"]) * (((((((data["ffti_ave_roll_mean_10000"]) + (data["fftr_percentile_40"]))) + (((data["fftr_autocorrelation_1000"]) * (data["exp_Moving_average_30000_std"]))))) + ((((data["fftr_min_roll_std_500"]) + (data["fftr_percentile_40"]))/2.0)))))) +

            0.039953*np.tanh((((((((data["fftr_min_roll_std_500"]) + (((data["fftr_min_roll_std_500"]) - (data["fftr_autocorrelation_1000"]))))/2.0)) + (((((data["spkt_welch_density_100"]) * (((data["fftr_min_roll_std_500"]) - (data["fftr_autocorrelation_1000"]))))) * 2.0)))) * 2.0)) +

            0.040000*np.tanh((((-1.0*((((data["ffti_ave10"]) * (((data["ffti_abs_percentile_1"]) + (((data["av_change_abs_roll_mean_500"]) + (data["exp_Moving_average_50000_std"])))))))))) - (((data["fftr_trend"]) * (((data["ffti_max_to_min"]) * 2.0)))))) +

            0.039937*np.tanh(((((((data["fftr_percentile_roll_std_60"]) + (((((((data["ffti_percentile_roll_std_75"]) * (((data["fftr_percentile_40"]) + (data["classic_sta_lta7_mean"]))))) + ((-1.0*((data["ffti_percentile_roll_std_75"])))))) * 2.0)))) * 2.0)) * 2.0)) +

            0.033654*np.tanh(((((((data["skew"]) + (data["ffti_exp_Moving_average_300_std"]))) + (((((((((((data["fftr_range_-2000_-1000"]) * (data["ffti_autocorrelation_5000"]))) * 2.0)) + (data["ffti_exp_Moving_average_300_std"]))) * 2.0)) * 2.0)))) * 2.0)) +

            0.039922*np.tanh((-1.0*((((data["ffti_min_roll_std_50"]) + (((data["max_last_10000"]) + ((((((data["fftr_std_roll_mean_10000"]) - (data["ffti_min_roll_std_500"]))) + ((((data["percentile_roll_mean_90"]) + (((data["ffti_autocorrelation_10"]) / 2.0)))/2.0)))/2.0))))))))) +

            0.040000*np.tanh((((((data["min_first_1000"]) * (data["fftr_min_roll_mean_10000"]))) + (((((data["ffti_autocorrelation_10000"]) * (((((((data["ffti_autocorrelation_10000"]) * (data["fftr_min_roll_std_10000"]))) * 2.0)) * (data["ffti_autocorrelation_10000"]))))) * 2.0)))/2.0)) +

            0.040000*np.tanh(((((data["fftr_percentile_roll_mean_70"]) * (((data["ffti_abs_percentile_1"]) - (((data["ffti_abs_max_roll_mean_1000"]) * (((data["ffti_abs_percentile_1"]) - (data["abs_percentile_90"]))))))))) - (((data["abs_percentile_90"]) - (data["ffti_abs_percentile_1"]))))) +

            0.040000*np.tanh(((data["ffti_percentile_roll_mean_90"]) * (((((data["mean_change_rate_last_1000"]) + (data["ffti_exp_Moving_average_300_mean"]))) + ((((-1.0*((((((data["mean_change_rate_last_1000"]) - (data["fftr_min_first_1000"]))) - (data["fftr_percentile_roll_mean_5"])))))) * 2.0)))))) +

            0.040000*np.tanh(((data["max_first_10000"]) + ((((data["skew"]) + (((((data["fftr_percentile_roll_mean_95"]) * ((-1.0*((((data["fftr_percentile_roll_std_40"]) + (data["mean_first_1000"])))))))) + (data["ffti_max_to_min"]))))/2.0)))) +

            0.040000*np.tanh(((((data["percentile_80"]) * (((data["fftr_classic_sta_lta1_mean"]) + ((((((data["classic_sta_lta4_mean"]) * (data["percentile_80"]))) + (((data["ffti_min_first_1000"]) * (((data["ffti_classic_sta_lta6_mean"]) * 2.0)))))/2.0)))))) * 2.0)) +

            0.040000*np.tanh(((((((((data["abs_max_roll_std_10"]) * (((((data["fftr_percentile_roll_std_10"]) - ((((((data["fftr_abs_max_roll_std_100"]) * 2.0)) + (data["ffti_ave10"]))/2.0)))) - (data["ffti_abs_max_roll_std_500"]))))) * 2.0)) * 2.0)) * 2.0)) +

            0.039984*np.tanh((-1.0*((((((((((((data["fftr_ave_roll_mean_50"]) + (data["ffti_ave10"]))/2.0)) + (data["av_change_abs_roll_mean_50"]))) + (data["fftr_max_roll_mean_100"]))/2.0)) + (((data["fftr_ave_roll_mean_50"]) * (((data["ffti_percentile_roll_std_10"]) * 2.0))))))))) +

            0.040000*np.tanh((((((data["fftr_mean_first_50000"]) + (data["av_change_abs_roll_mean_500"]))/2.0)) + (((data["fftr_skew"]) * (((((data["percentile_roll_std_40"]) * (((data["fftr_mean_first_50000"]) - (((data["ffti_percentile_roll_mean_30"]) / 2.0)))))) * 2.0)))))) +

            0.039984*np.tanh(((((data["num_peaks_50"]) * (data["ffti_min_roll_std_10000"]))) - (((((data["ffti_percentile_roll_mean_95"]) + (data["fftr_mean_last_50000"]))) * (((data["mean_change_rate_last_1000"]) + (((data["num_peaks_50"]) - (data["ffti_percentile_roll_mean_95"]))))))))) +

            0.040000*np.tanh(((((((((data["abs_max_roll_mean_100"]) + (((data["ffti_sum"]) * (((data["percentile_roll_std_60"]) + (((((data["ffti_av_change_abs_roll_std_500"]) + (data["ffti_av_change_abs_roll_std_500"]))) * 2.0)))))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["fftr_max_last_10000"]) * 2.0)) * ((((((data["abs_percentile_30"]) + (((data["ffti_mean_change_rate"]) - (data["fftr_min_roll_std_10"]))))/2.0)) - (data["ffti_mean_first_10000"]))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["av_change_rate_roll_std_50"]) * (((data["ffti_percentile_roll_mean_1"]) + (((((data["ffti_percentile_roll_mean_10"]) + (((data["ffti_ave_roll_mean_1000"]) + (data["fftr_autocorrelation_1000"]))))) + (data["max_to_min"]))))))) * 2.0)) * 2.0)) +

            0.039984*np.tanh(((((((data["ffti_av_change_abs_roll_mean_10"]) * (data["ffti_kstat_1"]))) + ((((((data["ffti_percentile_roll_mean_40"]) * (data["fftr_mean_last_1000"]))) + (data["skew"]))/2.0)))) + (((data["skew"]) * (data["mean_last_1000"]))))) +

            0.040000*np.tanh(((((data["ffti_mean_first_10000"]) * ((-1.0*((((data["fftr_min_first_1000"]) - ((((data["fftr_spkt_welch_density_1"]) + (((data["fftr_num_peaks_50"]) * (((data["ffti_mean_first_10000"]) + (data["ffti_classic_sta_lta8_mean"]))))))/2.0))))))))) * 2.0)) +

            0.040000*np.tanh(((data["ffti_percentile_roll_mean_60"]) * (((data["percentile_75"]) + (((data["fftr_abs_percentile_1"]) - (((((((((data["ffti_spkt_welch_density_5"]) * 2.0)) * 2.0)) - (data["av_change_abs_roll_std_50"]))) * 2.0)))))))) +

            0.039766*np.tanh(((data["ffti_av_change_abs_roll_std_10"]) * (((data["ffti_autocorrelation_1000"]) - (((((((data["ffti_max_roll_mean_1000"]) + (data["av_change_abs_roll_mean_500"]))) * 2.0)) - (data["ffti_autocorrelation_1000"]))))))) +

            0.040000*np.tanh(((data["av_change_abs_roll_mean_500"]) * (((data["percentile_70"]) + (((data["percentile_70"]) + (((((((data["spkt_welch_density_100"]) + (data["fftr_time_rev_asym_stat_1"]))) + (data["ffti_abs_percentile_10"]))) + (data["av_change_abs_roll_std_50"]))))))))) +

            0.040000*np.tanh(((((data["ffti_std_roll_mean_10000"]) * (((data["fftr_min_roll_std_10"]) - (((data["ffti_percentile_roll_std_50"]) - (data["min_roll_std_50"]))))))) - (((data["ffti_av_change_abs_roll_std_10"]) * (((data["ffti_std_roll_mean_10000"]) - (data["min_roll_std_50"]))))))) +

            0.040000*np.tanh(((data["ffti_num_peaks_100"]) * ((((((((data["ffti_num_peaks_100"]) + (data["fftr_percentile_roll_mean_70"]))/2.0)) * (data["fftr_time_rev_asym_stat_10"]))) - (((data["fftr_percentile_roll_mean_70"]) + (((data["fftr_num_peaks_20"]) * (data["abs_percentile_20"]))))))))) +

            0.039984*np.tanh((((-1.0*((data["av_change_abs_roll_std_500"])))) - (((((data["fftr_abs_max"]) * 2.0)) * (((data["ffti_min_roll_std_500"]) + (((data["ffti_autocorrelation_500"]) * (data["med"]))))))))) +

            0.039984*np.tanh(((data["ffti_num_peaks_20"]) * (((data["ffti_range_2000_3000"]) + (((data["fftr_autocorrelation_5"]) + (((data["ffti_range_2000_3000"]) + (((data["fftr_mean_last_1000"]) + (((data["av_change_abs_roll_std_10000"]) + (data["ffti_range_1000_2000"]))))))))))))) +

            0.038796*np.tanh(((((data["ffti_autocorrelation_1000"]) * (((((((((data["max_first_50000"]) + (data["fftr_mean_last_1000"]))) + (data["abs_percentile_20"]))) + (((data["percentile_roll_std_99"]) + (data["abs_percentile_20"]))))) * 2.0)))) * 2.0)) +

            0.040000*np.tanh(((((((((data["percentile_roll_std_1"]) - ((((data["mean_change_rate_first_1000"]) + (data["av_change_abs_roll_mean_10000"]))/2.0)))) - (data["ffti_av_change_abs_roll_std_50"]))) - (((data["fftr_mean_first_50000"]) + (data["min_roll_std_10"]))))) * (data["ffti_av_change_abs_roll_std_50"]))) +

            0.040000*np.tanh(((((data["ffti_c3_100"]) + ((((((((data["min_roll_std_50"]) + (data["ffti_abs_max_roll_mean_1000"]))/2.0)) + (data["ffti_spkt_welch_density_1"]))) + ((((data["ffti_spkt_welch_density_1"]) + (data["mean_last_1000"]))/2.0)))))) * (data["spkt_welch_density_50"]))) +

            0.037421*np.tanh(((data["fftr_percentile_roll_mean_70"]) * ((((((((((((((((((data["ffti_num_peaks_10"]) * 2.0)) * 2.0)) + (data["fftr_num_peaks_10"]))/2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) + (data["fftr_percentile_roll_mean_70"]))))) +

            0.040000*np.tanh(((((data["fftr_time_rev_asym_stat_1"]) * (data["classic_sta_lta2_mean"]))) + (((data["std_first_1000"]) * (((data["fftr_abs_max_roll_mean_10000"]) + ((-1.0*((data["fftr_time_rev_asym_stat_100"])))))))))) +

            0.040000*np.tanh(((data["ffti_percentile_roll_std_5"]) * (((((((data["fftr_percentile_roll_std_50"]) + (data["ffti_ave_roll_mean_10000"]))) + (data["ffti_kurt"]))) + (((((data["ffti_percentile_roll_mean_10"]) - (data["ffti_spkt_welch_density_10"]))) + (data["ffti_percentile_roll_mean_10"]))))))) +

            0.040000*np.tanh(((data["min_roll_std_100"]) * (((((data["min_roll_std_500"]) + ((((((data["fftr_time_rev_asym_stat_100"]) + (data["autocorrelation_5000"]))) + (data["ffti_av_change_abs_roll_std_10000"]))/2.0)))) + ((-1.0*((data["percentile_50"])))))))) +

            0.040000*np.tanh(((data["min_roll_mean_500"]) + ((((data["min_roll_mean_500"]) + (((data["ffti_av_change_abs_roll_std_10000"]) * (((data["ffti_range_p4000_pinf"]) - (((((data["ffti_av_change_abs_roll_std_10000"]) - (data["ffti_av_change_abs_roll_std_50"]))) - (data["ffti_av_change_abs_roll_std_50"]))))))))/2.0)))) +

            0.040000*np.tanh(((data["fftr_Hann_window_mean_50"]) * ((((data["ffti_range_2000_3000"]) + (((((data["ffti_percentile_roll_mean_5"]) * (((((((data["ffti_abs_max_roll_std_10000"]) * 2.0)) + (data["ffti_mean_change_rate_last_10000"]))) * 2.0)))) * 2.0)))/2.0)))) +

            0.039969*np.tanh((((((((data["ffti_exp_Moving_average_300_mean"]) * (data["fftr_spkt_welch_density_1"]))) - (data["fftr_mean_change_rate_last_1000"]))) + ((((((((data["ffti_percentile_roll_std_40"]) + (data["ffti_min_roll_mean_10000"]))/2.0)) - (data["fftr_mean_change_rate_last_1000"]))) - (data["fftr_num_peaks_10"]))))/2.0)) +

            0.040000*np.tanh(((data["fftr_min_roll_mean_100"]) * (((data["std_roll_mean_1000"]) + (((((data["fftr_percentile_roll_std_1"]) + (((data["ffti_autocorrelation_50"]) + (((data["ffti_autocorrelation_50"]) * (data["med"]))))))) * 2.0)))))) +

            0.035514*np.tanh(((data["ffti_mean_last_1000"]) * (((((data["ffti_percentile_roll_std_1"]) + (((data["ffti_av_change_abs_roll_std_50"]) + (((data["ffti_max_roll_mean_500"]) + ((((data["classic_sta_lta5_mean"]) + (data["ffti_percentile_roll_mean_80"]))/2.0)))))))) + (data["fftr_mean_change_rate_first_10000"]))))) +

            0.040000*np.tanh(((data["ffti_av_change_abs_roll_std_10000"]) * (((data["kurt"]) - ((((((((data["ffti_av_change_abs_roll_mean_50"]) + (data["fftr_abs_percentile_20"]))/2.0)) - (data["ffti_abs_percentile_5"]))) + (((data["ffti_exp_Moving_std_50000_std"]) * (data["kurt"]))))))))) +

            0.040000*np.tanh(((data["fftr_abs_percentile_70"]) * (((data["fftr_min_first_1000"]) + (((((((data["fftr_percentile_roll_std_40"]) * (data["av_change_abs_roll_std_50"]))) - (data["fftr_range_2000_3000"]))) - (data["fftr_mean_change_rate_first_1000"]))))))) +

            0.040000*np.tanh(((((data["percentile_25"]) - (data["av_change_abs_roll_std_500"]))) * (((data["ffti_max_to_min"]) - ((-1.0*(((((((data["ffti_mean_change_rate"]) + (data["av_change_abs_roll_std_500"]))/2.0)) * (data["percentile_25"])))))))))) +

            0.040000*np.tanh(((data["fftr_time_rev_asym_stat_5"]) * (((((data["ffti_percentile_roll_std_30"]) - (((data["ffti_av_change_abs_roll_mean_10"]) * (data["std_first_1000"]))))) - (((data["std_first_1000"]) * (data["std_first_1000"]))))))) +

            0.040000*np.tanh(((data["percentile_roll_mean_50"]) * (((data["ffti_max_to_min"]) * (((((data["ffti_std"]) + (data["mean_first_10000"]))) - (((data["abs_percentile_25"]) - (((data["mean_change_rate_first_10000"]) - (data["ffti_percentile_roll_mean_75"]))))))))))) +

            0.039969*np.tanh(((data["min_roll_std_10"]) * ((-1.0*((((data["iqr"]) - (((data["exp_Moving_average_50000_std"]) - (((data["fftr_classic_sta_lta6_mean"]) * (data["exp_Moving_average_50000_std"])))))))))))) +

            0.039953*np.tanh(((data["ffti_range_-3000_-2000"]) * ((((data["fftr_Hann_window_mean_50"]) + ((((-1.0*((((data["mean_change_rate_first_50000"]) * (((((data["fftr_av_change_abs_roll_mean_10000"]) * 2.0)) * 2.0))))))) + (data["ffti_percentile_roll_std_10"]))))/2.0)))) +

            0.040000*np.tanh(((((data["ffti_exp_Moving_std_3000_std"]) + (((data["fftr_autocorrelation_1000"]) + ((((data["fftr_max_to_min"]) + (((data["min_roll_std_1000"]) + (data["fftr_max_to_min"]))))/2.0)))))) * (((data["ffti_exp_Moving_std_3000_std"]) - (data["abs_max_roll_mean_10000"]))))) +

            0.039562*np.tanh(((data["fftr_std_roll_mean_10"]) * (((((((((data["fftr_time_rev_asym_stat_1"]) + (data["fftr_percentile_roll_mean_70"]))) + (((data["fftr_time_rev_asym_stat_1"]) + (data["fftr_percentile_roll_mean_70"]))))) + (data["autocorrelation_1000"]))) + (data["fftr_mean_change_rate_last_1000"]))))) +

            0.040000*np.tanh(((data["fftr_std_roll_mean_100"]) * (((data["fftr_percentile_roll_mean_30"]) * (((((((((data["fftr_abs_mean"]) * (data["fftr_c3_5"]))) - (data["ffti_percentile_roll_mean_90"]))) * (data["min_roll_std_10"]))) - (data["ffti_percentile_roll_mean_99"]))))))) +

            0.034670*np.tanh(((data["fftr_spkt_welch_density_10"]) * (((((((((data["fftr_percentile_roll_std_10"]) + (((data["av_change_abs_roll_mean_10000"]) + (((((data["av_change_abs_roll_std_500"]) + (data["ffti_min_roll_std_10"]))) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)))) +

            0.040000*np.tanh((-1.0*((((data["fftr_percentile_roll_std_30"]) * ((((((-1.0*((data["autocorrelation_100"])))) + (data["fftr_ave_roll_mean_50"]))) + (((data["fftr_percentile_roll_std_30"]) + (data["ffti_autocorrelation_5000"])))))))))) +

            0.040000*np.tanh((((((data["percentile_roll_std_20"]) * (((data["percentile_roll_std_20"]) * (data["ffti_min_roll_std_10"]))))) + ((((((((data["min_roll_std_500"]) + (data["ffti_min_roll_std_10"]))/2.0)) * (data["fftr_time_rev_asym_stat_1"]))) - (data["ffti_num_peaks_50"]))))/2.0)) +

            0.040000*np.tanh(((data["av_change_rate_roll_std_1000"]) * (((data["fftr_hmean"]) - (((((data["fftr_hmean"]) * (data["fftr_hmean"]))) + ((((-1.0*((((data["ffti_percentile_50"]) * 2.0))))) * 2.0)))))))) +

            0.039969*np.tanh(((((((((data["ffti_percentile_roll_mean_10"]) * (((data["ffti_min_roll_std_500"]) + (((data["fftr_min_roll_std_100"]) * (data["fftr_min_roll_std_50"]))))))) + (data["min_roll_std_10"]))/2.0)) + (((data["fftr_min_roll_std_100"]) * (data["fftr_min_roll_std_50"]))))/2.0)) +

            0.040000*np.tanh((((((((((data["ffti_autocorrelation_5000"]) + (data["ffti_min_roll_std_100"]))/2.0)) + (((data["ffti_autocorrelation_5000"]) - (data["ffti_percentile_roll_std_40"]))))) * (data["trend"]))) - (((data["ffti_percentile_roll_mean_25"]) * (data["fftr_abs_percentile_1"]))))) +

            0.039984*np.tanh(((data["mean_change_rate_first_1000"]) * (((((data["spkt_welch_density_100"]) + (((data["ave_roll_mean_50"]) + (data["fftr_percentile_roll_std_25"]))))) * (((data["fftr_percentile_roll_std_1"]) - (((data["std_roll_mean_100"]) / 2.0)))))))) +

            0.040000*np.tanh(((data["ffti_min_first_10000"]) * (((data["ffti_std_roll_std_100"]) - (((data["ffti_percentile_roll_std_10"]) + (((((((data["percentile_70"]) - (data["fftr_abs_percentile_5"]))) + (data["ffti_mean_first_50000"]))) + (data["fftr_hmean"]))))))))) +

            0.039984*np.tanh(((data["fftr_mean_first_50000"]) * (((data["ffti_mean_change_rate_last_10000"]) + (((data["ffti_mean_change_rate_last_10000"]) + (((((data["fftr_min_roll_mean_10"]) * (data["num_peaks_10"]))) + ((((data["ffti_std_roll_mean_1000"]) + (data["max_to_min"]))/2.0)))))))))) +

            0.039969*np.tanh((-1.0*((((data["ffti_med"]) * (((((((data["max_first_1000"]) + ((-1.0*((((data["ffti_med"]) * (data["av_change_abs_roll_mean_500"])))))))) + (data["av_change_abs_roll_mean_10"]))) + (data["max_first_1000"])))))))) +

            0.040000*np.tanh((((((((data["fftr_percentile_roll_std_40"]) + ((((((data["ffti_max_to_min"]) * (data["ffti_max_last_1000"]))) + (data["ffti_percentile_roll_mean_10"]))/2.0)))) + (((data["ffti_percentile_roll_mean_60"]) * (data["fftr_min_roll_std_100"]))))/2.0)) - (data["fftr_mean_change_rate_first_10000"]))) +

            0.040000*np.tanh(((((((data["fftr_time_rev_asym_stat_5"]) * (((data["ffti_percentile_roll_std_60"]) * (((data["ffti_ave_roll_mean_10000"]) - (((((((data["ffti_med"]) + (data["fftr_time_rev_asym_stat_5"]))/2.0)) + (data["fftr_time_rev_asym_stat_10"]))/2.0)))))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((data["ffti_percentile_roll_std_30"]) * (((((((data["autocorrelation_500"]) * (data["autocorrelation_500"]))) - (((data["fftr_autocorrelation_500"]) - (data["max_to_min_diff"]))))) - (((data["fftr_autocorrelation_500"]) - (data["fftr_time_rev_asym_stat_50"]))))))) +

            0.039984*np.tanh((((((((((data["med"]) * (((((data["ffti_percentile_roll_std_40"]) / 2.0)) - (data["ffti_percentile_roll_std_60"]))))) - (data["fftr_mean_change_rate_first_10000"]))) * 2.0)) + (((data["ffti_percentile_roll_std_40"]) - (data["fftr_percentile_roll_std_5"]))))/2.0)) +

            0.040000*np.tanh((((((data["fftr_abs_percentile_1"]) * (data["fftr_autocorrelation_100"]))) + ((-1.0*((((data["fftr_time_rev_asym_stat_50"]) * (((data["ffti_hmean"]) + (((data["ffti_Hann_window_mean_1500"]) + (data["exp_Moving_average_50000_std"])))))))))))/2.0)) +

            0.039969*np.tanh(((((data["fftr_c3_1000"]) - (((data["fftr_mean_change_rate_first_10000"]) + (data["fftr_mean_change_rate_first_10000"]))))) - (((((((data["ffti_av_change_abs_roll_std_50"]) + (data["av_change_abs_roll_std_10000"]))/2.0)) + ((((data["ffti_av_change_abs_roll_std_10000"]) + (data["ffti_ave10"]))/2.0)))/2.0)))) +

            0.039969*np.tanh(((((((data["fftr_kstat_1"]) * (data["ffti_mean_change_rate_last_10000"]))) + (((data["abs_percentile_20"]) * ((-1.0*((((((data["fftr_time_rev_asym_stat_10"]) / 2.0)) + (data["num_peaks_100"])))))))))) - (data["fftr_mean_change_rate_last_1000"]))) +

            0.040000*np.tanh((((((data["std_first_10000"]) + (data["fftr_min_roll_std_500"]))/2.0)) * (((data["fftr_min_roll_std_10"]) + (((data["fftr_min_roll_std_10"]) + (((data["ffti_percentile_roll_std_10"]) + ((((data["ffti_percentile_roll_std_10"]) + (data["percentile_roll_mean_95"]))/2.0)))))))))) +

            0.039969*np.tanh(((data["ffti_range_-3000_-2000"]) * ((((((data["ffti_av_change_abs_roll_std_10000"]) + (data["fftr_min_roll_std_50"]))/2.0)) + (((data["fftr_percentile_roll_mean_5"]) + ((((-1.0*((data["ffti_percentile_roll_std_25"])))) + (data["fftr_percentile_roll_std_10"]))))))))) +

            0.039984*np.tanh((((((((((data["fftr_range_2000_3000"]) + (data["min_first_1000"]))) + (data["percentile_roll_mean_10"]))) * (data["fftr_time_rev_asym_stat_5"]))) + (((data["min_first_1000"]) * (((data["ffti_mean_first_10000"]) + (data["fftr_num_peaks_50"]))))))/2.0)) +

            0.040000*np.tanh(((((data["percentile_roll_mean_1"]) + ((((((data["fftr_time_rev_asym_stat_100"]) + (data["num_peaks_100"]))/2.0)) * (data["num_peaks_100"]))))) * (((data["fftr_iqr1"]) - ((((data["fftr_time_rev_asym_stat_100"]) + (data["fftr_min_roll_std_1000"]))/2.0)))))) +

            0.035889*np.tanh((((((((((data["classic_sta_lta6_mean"]) + (data["fftr_percentile_20"]))/2.0)) * 2.0)) + (((data["skew"]) * (data["fftr_percentile_30"]))))) + (((((data["skew"]) * (data["fftr_autocorrelation_500"]))) * 2.0)))) +

            0.039969*np.tanh(((((data["ffti_spkt_welch_density_5"]) * (((data["ffti_ave_roll_mean_10"]) + (data["ffti_ave_roll_mean_10"]))))) + (((data["ffti_kstat_2"]) - (((data["fftr_spkt_welch_density_5"]) + ((((data["fftr_spkt_welch_density_5"]) + (data["ffti_abs_percentile_5"]))/2.0)))))))) +

            0.039969*np.tanh((((((((((data["percentile_roll_std_80"]) + (data["min_last_10000"]))/2.0)) + (data["ffti_autocorrelation_5"]))/2.0)) + (((data["ffti_ave_roll_mean_10000"]) * (((data["fftr_time_rev_asym_stat_1"]) - (data["percentile_roll_std_80"]))))))/2.0)) +

            0.039984*np.tanh(((((((data["av_change_rate_roll_mean_50"]) - (data["autocorrelation_1000"]))) * (data["abs_max_roll_std_10"]))) + (((data["ffti_percentile_roll_std_20"]) + (((data["ffti_min_roll_std_100"]) * (((data["ffti_percentile_roll_std_20"]) + (data["av_change_rate_roll_mean_100"]))))))))) +

            0.040000*np.tanh((((((((data["fftr_min_roll_std_500"]) + (((data["fftr_mean_change_rate"]) - (data["ffti_percentile_roll_std_20"]))))/2.0)) - (((data["ffti_percentile_roll_std_20"]) * (data["fftr_min_roll_std_500"]))))) - (((data["fftr_min_last_1000"]) * (data["fftr_exp_Moving_average_300_mean"]))))) +

            0.039984*np.tanh(((((data["std_roll_mean_10000"]) * (((data["ffti_av_change_abs_roll_mean_1000"]) * (((data["fftr_ave_roll_mean_10000"]) + ((((-1.0*((data["max_to_min"])))) * (data["fftr_ave_roll_mean_10000"]))))))))) * 2.0)) +

            0.040000*np.tanh(((((data["ffti_av_change_abs_roll_std_50"]) * (((data["ffti_spkt_welch_density_1"]) - ((((data["fftr_sum"]) + (((data["abs_percentile_20"]) * ((-1.0*((((data["percentile_roll_mean_30"]) + (data["classic_sta_lta4_mean"])))))))))/2.0)))))) * 2.0)) +

            0.039969*np.tanh(((((((data["fftr_time_rev_asym_stat_100"]) - (((data["fftr_ave_roll_mean_10000"]) - (((data["ffti_percentile_roll_std_25"]) - (data["ffti_percentile_roll_mean_75"]))))))) * (((data["ffti_percentile_roll_std_25"]) - (data["ffti_mean_change_rate_last_1000"]))))) * 2.0)) +

            0.039015*np.tanh((((-1.0*((((((data["fftr_percentile_roll_mean_1"]) - (data["min_roll_mean_1000"]))) - (((data["fftr_autocorrelation_100"]) * (((data["ffti_spkt_welch_density_1"]) + (data["fftr_percentile_roll_mean_1"])))))))))) * 2.0)) +

            0.039984*np.tanh((((((((data["fftr_autocorrelation_10000"]) + (((data["ffti_kstat_1"]) * (((data["fftr_autocorrelation_10000"]) * (data["mean_change_rate_first_1000"]))))))/2.0)) + (data["ffti_med"]))) * (((data["percentile_roll_std_90"]) + (data["mean_change_rate_first_1000"]))))) +

            0.040000*np.tanh(((((((data["ffti_av_change_abs_roll_std_10000"]) + (data["fftr_percentile_roll_mean_1"]))) * ((((data["ffti_abs_percentile_1"]) + ((((data["fftr_range_-3000_-2000"]) + ((-1.0*(((-1.0*((data["fftr_c3_1000"]))))))))/2.0)))/2.0)))) + (data["fftr_c3_50"]))) +

            0.040000*np.tanh(((data["ffti_kstat_1"]) * (((((((data["autocorrelation_1000"]) + (((data["ffti_kstat_1"]) * (((data["autocorrelation_1000"]) + (data["ffti_ave_roll_mean_10000"]))))))) + (data["ffti_percentile_roll_mean_70"]))) + (data["ffti_percentile_roll_mean_70"]))))) +

            0.039969*np.tanh(((data["av_change_abs_roll_std_1000"]) * (((((((data["min_last_1000"]) * (((((data["min_last_1000"]) * (data["fftr_max_last_50000"]))) * (data["min_last_1000"]))))) - (data["ave_roll_mean_10000"]))) * (data["min_last_1000"]))))) +

            0.040000*np.tanh(((data["ffti_min_roll_std_100"]) * (((((((data["ffti_min_roll_std_100"]) + ((-1.0*((data["ffti_min_roll_std_50"])))))/2.0)) + ((((-1.0*((data["ffti_min_roll_std_500"])))) - (data["av_change_abs_roll_mean_100"]))))/2.0)))) +

            0.039969*np.tanh(((((data["min_first_1000"]) * (data["max_roll_mean_10000"]))) * (((((-1.0*((((data["ffti_autocorrelation_500"]) - ((((data["ave_roll_mean_500"]) + (data["percentile_roll_mean_5"]))/2.0))))))) + ((-1.0*((data["ffti_autocorrelation_100"])))))/2.0)))) +

            0.039984*np.tanh(((((data["ffti_max_roll_mean_500"]) * ((-1.0*(((((data["percentile_25"]) + ((((((data["abs_percentile_25"]) * (data["percentile_roll_std_25"]))) + ((((data["ffti_av_change_abs_roll_std_10"]) + (data["percentile_roll_std_20"]))/2.0)))/2.0)))/2.0))))))) * 2.0)) +

            0.040000*np.tanh(((((data["ffti_percentile_roll_std_10"]) * (((((((data["fftr_std"]) * 2.0)) + (((data["ffti_ave_roll_mean_10"]) * (data["ffti_av_change_abs_roll_std_50"]))))) * 2.0)))) - (((data["ffti_ave_roll_mean_10"]) * (data["ffti_av_change_abs_roll_std_50"]))))) +

            0.040000*np.tanh(((data["ffti_mean_first_10000"]) * ((((data["ffti_autocorrelation_100"]) + (((data["kstat_1"]) * (((data["ffti_autocorrelation_100"]) + (((data["ffti_mean_first_10000"]) * (((data["ffti_autocorrelation_100"]) - (data["ffti_percentile_roll_std_10"]))))))))))/2.0)))) +

            0.040000*np.tanh((((((-1.0*((((data["fftr_mean_change_rate_last_1000"]) - (((((data["ffti_mean_change_rate_last_10000"]) * (((data["fftr_mean_first_50000"]) - (((data["fftr_Hann_window_mean_150"]) * (data["ffti_mean_last_50000"]))))))) * 2.0))))))) * 2.0)) * 2.0)) +

            0.039969*np.tanh((((((data["fftr_percentile_roll_std_40"]) * ((-1.0*((data["fftr_min_roll_std_500"])))))) + (((((data["av_change_abs_roll_std_50"]) * (data["av_change_abs_roll_std_50"]))) * (((data["av_change_abs_roll_std_50"]) * (data["fftr_av_change_abs_roll_mean_100"]))))))/2.0)) +

            0.039984*np.tanh(((((((data["ffti_mean_change_rate"]) + (data["fftr_percentile_roll_std_10"]))) + (((data["fftr_max"]) * (((data["fftr_c3_50"]) + (data["fftr_percentile_roll_std_10"]))))))) * (((data["fftr_max"]) * (data["ffti_percentile_roll_std_25"]))))) +

            0.037233*np.tanh(((((((data["std_roll_mean_500"]) * (((((data["av_change_abs_roll_std_100"]) + (data["ffti_kstat_1"]))) + (((data["ffti_num_peaks_100"]) + ((((data["av_change_abs_roll_std_100"]) + (data["percentile_roll_mean_5"]))/2.0)))))))) * 2.0)) * 2.0)) +

            0.039984*np.tanh(((((data["ffti_min_roll_std_500"]) + (((data["autocorrelation_50"]) * (((((data["ffti_exp_Moving_average_30000_mean"]) * (data["ffti_min_roll_std_500"]))) - (data["fftr_ave_roll_mean_10"]))))))) * (data["ffti_exp_Moving_average_30000_mean"]))) +

            0.040000*np.tanh(((data["fftr_hmean"]) * (((((((data["classic_sta_lta4_mean"]) + (((data["av_change_abs_roll_mean_1000"]) + (data["autocorrelation_10000"]))))/2.0)) + (((data["av_change_abs_roll_std_100"]) - (((data["classic_sta_lta4_mean"]) * (data["av_change_abs_roll_mean_10000"]))))))/2.0)))) +

            0.039969*np.tanh(((data["std_last_50000"]) * (((data["fftr_autocorrelation_10"]) * (((data["fftr_classic_sta_lta6_mean"]) + (((data["fftr_autocorrelation_10"]) * (((data["fftr_autocorrelation_10"]) * (((data["fftr_autocorrelation_10"]) * (data["fftr_c3_5"]))))))))))))) +

            0.039922*np.tanh(((data["kstatvar_1"]) * (((((((data["fftr_percentile_roll_std_5"]) + (((((data["ffti_autocorrelation_1000"]) + (data["ffti_Hann_window_mean_15000"]))) + (((data["std_first_10000"]) * (data["kstatvar_1"]))))))) * 2.0)) * 2.0)))) +

            0.039984*np.tanh(((data["ffti_mean_change_rate_last_10000"]) * (((((((((((((((((((data["ffti_mean_change_rate_last_1000"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) + (data["ffti_sum"]))) * 2.0)) * 2.0)))) +

            0.036717*np.tanh(((data["av_change_abs_roll_mean_100"]) * (((data["ffti_av_change_abs_roll_std_1000"]) + (((((data["fftr_percentile_roll_std_5"]) * (data["fftr_autocorrelation_100"]))) + (((data["fftr_percentile_roll_std_40"]) + (data["fftr_percentile_roll_std_5"]))))))))) +

            0.039922*np.tanh(((data["fftr_hmean"]) * ((((((((data["ffti_max_to_min_diff"]) + (((data["fftr_hmean"]) * (data["fftr_exp_Moving_std_300_std"]))))/2.0)) + (data["av_change_rate_roll_std_1000"]))) + (((data["fftr_max_roll_std_10"]) * (data["ffti_min_last_1000"]))))))) +

            0.040000*np.tanh(((data["fftr_mean_change_rate_first_10000"]) * (((((data["fftr_percentile_roll_mean_20"]) - (((((data["mean_last_10000"]) + (((data["fftr_std_roll_mean_100"]) + (data["fftr_moment_3"]))))) * 2.0)))) - (((data["ffti_mean_last_10000"]) * 2.0)))))) +

            0.039859*np.tanh((((((data["ffti_spkt_welch_density_10"]) + (data["ffti_exp_Moving_std_300_mean"]))/2.0)) * (((((data["ffti_percentile_roll_mean_70"]) + (data["c3_5000"]))) * (((data["ffti_spkt_welch_density_10"]) + (((((data["ffti_ave_roll_mean_1000"]) * 2.0)) * 2.0)))))))) +

            0.039984*np.tanh((-1.0*((((data["ffti_Hann_window_mean_1500"]) * ((-1.0*((((data["autocorrelation_5000"]) * (((data["ffti_autocorrelation_10000"]) - (((data["ffti_Hann_window_mean_1500"]) * (((((data["ffti_Hann_window_mean_1500"]) / 2.0)) / 2.0)))))))))))))))) +

            0.040000*np.tanh(((data["ffti_std_roll_mean_1000"]) * (((data["ffti_mean_last_50000"]) * (((data["percentile_20"]) + (((((((data["fftr_exp_Moving_average_3000_mean"]) - (data["fftr_num_peaks_50"]))) + (data["ffti_av_change_abs_roll_std_500"]))) + (data["max_first_50000"]))))))))) +

            0.037483*np.tanh((-1.0*((((((((((data["ffti_av_change_abs_roll_std_500"]) + (data["fftr_percentile_roll_std_10"]))/2.0)) * 2.0)) + ((((data["percentile_roll_std_20"]) + (((data["fftr_autocorrelation_1000"]) * (((data["fftr_autocorrelation_1000"]) * (data["ffti_autocorrelation_500"]))))))/2.0)))/2.0))))) +

            0.036546*np.tanh(((((data["av_change_rate_roll_mean_100"]) * (((data["exp_Moving_average_30000_std"]) + (data["fftr_std_last_1000"]))))) + (((((data["fftr_mean_last_10000"]) * (data["ffti_percentile_roll_mean_80"]))) * ((-1.0*((((data["exp_Moving_average_30000_std"]) * 2.0))))))))) +

            0.040000*np.tanh(((data["fftr_autocorrelation_5000"]) * ((((((((((data["min_first_10000"]) * 2.0)) * 2.0)) - (((data["fftr_std_roll_mean_100"]) * (data["ffti_autocorrelation_100"]))))) + ((-1.0*((data["av_change_abs_roll_std_50"])))))/2.0)))) +

            0.039218*np.tanh(((data["ffti_ave_roll_std_500"]) * (((((((data["time_rev_asym_stat_10"]) - (((data["ffti_abs_percentile_1"]) * (((((data["ffti_autocorrelation_10000"]) + (data["ffti_ave_roll_std_50"]))) + (data["fftr_percentile_roll_std_10"]))))))) * 2.0)) * 2.0)))) +

            0.040000*np.tanh(((data["abs_max_roll_mean_50"]) + (((((((data["av_change_abs_roll_mean_500"]) + ((((data["fftr_percentile_roll_mean_99"]) + (data["mean_change_rate_first_1000"]))/2.0)))/2.0)) + (((data["min_first_1000"]) * (((data["ffti_num_peaks_50"]) - (data["fftr_percentile_roll_mean_99"]))))))/2.0)))) +

            0.040000*np.tanh(((data["iqr"]) * (((data["ffti_max_to_min"]) * ((((((data["trend"]) + (data["fftr_abs_percentile_90"]))/2.0)) + (((((data["fftr_abs_percentile_90"]) * 2.0)) + (data["fftr_ave_roll_mean_1000"]))))))))) +

            0.039953*np.tanh(((data["percentile_75"]) * ((-1.0*(((((((data["med"]) - (((data["ffti_av_change_abs_roll_std_10"]) * (data["ffti_mean_last_50000"]))))) + (((data["ffti_hmean"]) - (data["fftr_max_roll_std_100"]))))/2.0))))))) +

            0.036592*np.tanh(((((data["mean_change_rate_first_50000"]) * ((((data["fftr_max_roll_std_10000"]) + (data["ffti_autocorrelation_1000"]))/2.0)))) + ((-1.0*((((data["mean_change_rate_first_50000"]) * (((data["ffti_max_to_min_diff"]) * (((data["ffti_min_roll_std_50"]) * 2.0))))))))))) +

            0.039969*np.tanh(((((data["fftr_min_roll_std_100"]) * (((data["ffti_percentile_roll_std_40"]) + ((((data["abs_percentile_20"]) + (data["max_first_10000"]))/2.0)))))) - (((data["ffti_percentile_roll_std_40"]) * (((((data["av_change_rate_roll_mean_10"]) * 2.0)) * 2.0)))))) +

            0.039984*np.tanh(((data["ffti_percentile_roll_mean_75"]) * (((data["ffti_min_roll_std_50"]) * (((((-1.0*((((data["percentile_roll_mean_1"]) * 2.0))))) + (data["ffti_min_roll_std_50"]))/2.0)))))) +

            0.039937*np.tanh((-1.0*((((data["fftr_std_roll_mean_1000"]) * (((((data["ffti_mean_change_rate_last_1000"]) + (data["percentile_60"]))) + ((((((data["percentile_60"]) * (data["percentile_60"]))) + (data["ffti_max_to_min_diff"]))/2.0))))))))) +

            0.033341*np.tanh((((data["num_peaks_20"]) + (((data["max_to_min"]) * (((((data["ffti_min_roll_std_1000"]) - (((data["fftr_autocorrelation_5000"]) - (((data["max_to_min"]) - (data["fftr_autocorrelation_5000"]))))))) - (data["fftr_autocorrelation_5000"]))))))/2.0)) +

            0.024541*np.tanh(((((data["abs_percentile_75"]) * (data["autocorrelation_5000"]))) + ((((-1.0*((((data["autocorrelation_5000"]) * (data["percentile_40"])))))) * (((data["autocorrelation_5000"]) * (data["percentile_40"]))))))) +

            0.039953*np.tanh(((data["min_first_50000"]) * (((((data["fftr_max_last_10000"]) + (data["ffti_percentile_roll_mean_5"]))) - (((((data["fftr_c3_1000"]) * (data["ffti_std_roll_std_10"]))) - (((data["fftr_abs_trend"]) + (data["ffti_num_peaks_100"]))))))))) +

            0.039984*np.tanh(((data["abs_percentile_30"]) * (((((data["fftr_min_roll_std_10"]) + (data["ffti_skew"]))) + (((((((data["ffti_skew"]) + (data["ffti_percentile_10"]))/2.0)) + (((data["time_rev_asym_stat_50"]) * (data["ffti_percentile_10"]))))/2.0)))))))

def GPI(data):

    return (5.577521 +

            0.040000*np.tanh((((((((((((((-1.0*((((data["ffti_range_3000_4000"]) + (data["percentile_roll_std_10"])))))) - (data["num_peaks_10"]))) * 2.0)) - (data["percentile_roll_std_20"]))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((-1.0*((((((((data["fftr_range_-3000_-2000"]) + (((((data["percentile_roll_std_10"]) + (((data["percentile_95"]) + (data["ffti_range_-4000_-3000"]))))) * 2.0)))) * 2.0)) * 2.0))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((((((-1.0*((data["fftr_range_3000_4000"])))) - (data["percentile_roll_std_10"]))) - (data["num_peaks_10"]))) * 2.0)) - (data["iqr"]))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((-1.0*((((data["range_-1000_0"]) + (data["iqr"])))))) - (((data["ffti_range_2000_3000"]) + (data["percentile_roll_std_5"]))))) * 2.0)) - (data["min_roll_std_1000"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((14.20582294464111328)) * ((((((((-1.0*((((data["min_roll_std_10000"]) + (((data["percentile_roll_std_25"]) + (((data["ffti_range_3000_4000"]) * 2.0))))))))) * 2.0)) - (data["percentile_roll_std_5"]))) * 2.0)))) +

            0.040000*np.tanh((((-1.0*((((data["min_roll_std_100"]) + (((((data["iqr"]) + (((((data["num_peaks_10"]) + (((data["percentile_roll_std_5"]) + (data["ffti_range_-4000_-3000"]))))) * 2.0)))) * 2.0))))))) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["ffti_range_-1000_0"]) - (((data["num_peaks_10"]) + (data["fftr_range_-3000_-2000"]))))) - (data["percentile_95"]))) - (data["percentile_95"]))) * 2.0)) * 2.0)) - (data["min_roll_std_500"]))) +

            0.040000*np.tanh((((((((((-1.0*((((((data["range_-1000_0"]) + (data["num_peaks_10"]))) + (data["iqr"])))))) - (((data["ffti_range_3000_4000"]) + (data["percentile_roll_std_1"]))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((-2.0) - (data["fftr_range_2000_3000"]))) - (((((((data["percentile_roll_std_10"]) + (((data["percentile_95"]) * 2.0)))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((-1.0*((((((((data["min_roll_std_1000"]) + (((((((((data["num_peaks_10"]) + (data["percentile_roll_std_20"]))) + (data["fftr_range_-4000_-3000"]))) * 2.0)) + (data["iqr"]))))) * 2.0)) * 2.0))))) +

            0.040000*np.tanh(((((((data["range_0_1000"]) - (((((data["iqr"]) + (((data["range_-1000_0"]) + (((data["min_roll_std_500"]) + (data["abs_percentile_80"]))))))) + (data["fftr_range_-3000_-2000"]))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((-3.0) * (((data["fftr_range_2000_3000"]) + (((data["abs_percentile_80"]) + (((((data["num_peaks_10"]) + (data["percentile_roll_std_5"]))) * 2.0)))))))) * 2.0)) +

            0.040000*np.tanh((((((-1.0*((((((data["percentile_roll_std_20"]) + (((((data["ffti_range_-3000_-2000"]) + (((((data["gmean"]) + (data["range_-1000_0"]))) * 2.0)))) * 2.0)))) * 2.0))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["range_0_1000"]) - ((((((data["ffti_count_big"]) + (((((data["fftr_exp_Moving_average_30000_std"]) - (data["mean_change_rate"]))) * 2.0)))/2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["fftr_range_-1000_0"]) - (((((((data["iqr"]) + (((data["num_peaks_10"]) * 2.0)))) + (((data["percentile_roll_std_10"]) + (data["ffti_range_2000_3000"]))))) * 2.0)))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((data["fftr_av_change_abs_roll_mean_50"]) + (((data["range_0_1000"]) * 2.0)))) * 2.0)) - (data["fftr_range_-4000_-3000"]))) * 2.0)) * 2.0)) - (data["fftr_av_change_abs_roll_mean_50"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["ffti_range_0_1000"]) - ((((data["min_roll_std_100"]) + (data["iqr"]))/2.0)))) - (((data["fftr_range_-3000_-2000"]) + (data["percentile_roll_std_5"]))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((((data["range_0_1000"]) * 2.0)) - (data["num_peaks_10"]))) - (data["iqr"]))) * 2.0)) * 2.0)) * 2.0)) - (data["percentile_roll_mean_20"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((-1.0*((data["iqr"])))) - (((((((((data["iqr"]) + (((((data["percentile_roll_std_1"]) - (data["range_0_1000"]))) + (data["num_peaks_10"]))))) * 2.0)) * 2.0)) * 2.0)))) +

            0.040000*np.tanh(((data["num_peaks_10"]) - (((((((((((((data["num_peaks_10"]) + (((data["percentile_roll_std_25"]) + (data["ffti_range_-4000_-3000"]))))) + (data["gmean"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) +

            0.040000*np.tanh((((-1.0*((((((data["num_peaks_10"]) + (((((((((data["abs_percentile_80"]) + (data["percentile_roll_std_10"]))) + (data["fftr_range_-3000_-2000"]))) * 2.0)) + (data["min_roll_std_100"]))))) * 2.0))))) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((((data["mean_change_rate"]) - (data["iqr"]))) - (data["num_peaks_10"]))) * 2.0)) * 2.0)) - (data["min_roll_std_10000"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["mean_change_rate_last_50000"]) - (data["ave_roll_mean_100"]))) - (((data["ffti_range_3000_4000"]) + (data["num_peaks_10"]))))) * 2.0)) - (data["percentile_roll_std_1"]))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((((((-1.0*((data["ffti_range_-4000_-3000"])))) - (data["abs_percentile_80"]))) - ((((data["ffti_range_-3000_-2000"]) + (data["percentile_roll_std_5"]))/2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((-1.0*((data["num_peaks_10"])))) - (((((((data["num_peaks_10"]) + (data["iqr"]))) + (data["min_roll_std_10000"]))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["range_0_1000"]) - (((data["min_roll_std_10000"]) + (((((data["ffti_range_-4000_-3000"]) + (data["min_roll_std_500"]))) + (data["num_peaks_10"]))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["ffti_range_0_1000"]) - (data["abs_percentile_80"]))) - (data["ffti_range_2000_3000"]))) - (((data["ffti_abs_max_roll_mean_10"]) + (data["percentile_roll_std_5"]))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((-2.0) - (((((data["percentile_roll_std_1"]) + (((((data["iqr"]) - (((data["range_0_1000"]) * 2.0)))) + (data["num_peaks_10"]))))) * 2.0)))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["percentile_roll_std_5"]) - (((((((((data["percentile_roll_std_5"]) + (data["fftr_range_-4000_-3000"]))) + (((data["range_-1000_0"]) + (data["ave_roll_mean_1000"]))))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((-1.0*((((data["min_roll_std_500"]) + (((((((((data["percentile_roll_std_10"]) + (data["fftr_iqr"]))) + (data["abs_percentile_80"]))) * 2.0)) + (data["min_roll_std_50"])))))))) * 2.0)) +

            0.040000*np.tanh(((((((((data["ffti_range_-1000_0"]) + (((((((-1.0) - (((data["num_peaks_10"]) + (((data["percentile_roll_std_10"]) * 2.0)))))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["range_0_1000"]) - (data["percentile_75"]))) * 2.0)) - (((data["ffti_range_-3000_-2000"]) + (data["percentile_roll_std_5"]))))) * 2.0)) * 2.0)) - (data["percentile_roll_std_5"]))) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["ffti_range_0_1000"]) - (data["num_peaks_10"]))) - (data["percentile_roll_std_5"]))) - (((data["abs_percentile_80"]) * 2.0)))) * 2.0)) * 2.0)) - (data["ffti_range_-3000_-2000"]))) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["mean_change_rate_last_50000"]) - (((data["min_roll_std_50"]) - (((((data["ffti_range_-1000_0"]) - (data["num_peaks_10"]))) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["range_0_1000"]) - (((((1.0) + (((data["min_roll_std_1000"]) + (((data["percentile_roll_std_1"]) + (data["num_peaks_10"]))))))) * 2.0)))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((((-1.0*((((((data["percentile_roll_std_10"]) + (data["ffti_range_3000_4000"]))) + (((data["abs_percentile_80"]) * 2.0))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((data["mean_change_rate"]) - (((data["abs_percentile_50"]) + ((((data["min_roll_std_1000"]) + (data["num_peaks_10"]))/2.0)))))) - (((data["min_roll_std_100"]) + (data["num_peaks_10"]))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((-1.0*((((((((data["num_peaks_10"]) + (((((((data["percentile_roll_std_10"]) + (data["abs_percentile_80"]))) * 2.0)) * 2.0)))) + (data["fftr_abs_percentile_20"]))) * 2.0))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((((((-1.0*((data["num_peaks_10"])))) - (((data["range_-1000_0"]) * 2.0)))) - (((data["fftr_range_-3000_-2000"]) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["range_0_1000"]) * 2.0)) - (((data["min_roll_std_500"]) + (data["percentile_roll_std_10"]))))) - (((data["num_peaks_10"]) * 2.0)))) + (-2.0))) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["ffti_classic_sta_lta2_mean"]) - (((((((((data["abs_percentile_90"]) * 2.0)) + (data["iqr"]))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["range_0_1000"]) * 2.0)) - (((((((data["min_roll_std_500"]) + (data["min_roll_std_100"]))) + (((((data["min_roll_std_10000"]) + (data["abs_percentile_50"]))) * 2.0)))) * 2.0)))) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((data["mean_change_rate_last_50000"]) - (data["num_peaks_10"]))) * 2.0)) - (data["percentile_roll_std_40"]))) - (data["percentile_roll_std_1"]))) * 2.0)) - (data["percentile_roll_std_40"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((data["fftr_av_change_abs_roll_mean_10"]) - (((((((((((data["ffti_range_2000_3000"]) + (((((data["abs_percentile_90"]) * 2.0)) * 2.0)))) * 2.0)) + (data["min_roll_std_50"]))) * 2.0)) * 2.0)))) +

            0.040000*np.tanh(((((((((((((((((((data["fftr_av_change_abs_roll_std_10"]) + (((data["mean_change_rate"]) * 2.0)))) * 2.0)) * 2.0)) - (data["min_roll_std_500"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((((-1.0*((data["abs_percentile_25"])))) - (((((data["percentile_95"]) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((-1.0*((((data["min_roll_std_10000"]) + (data["percentile_roll_std_10"])))))) - (((((data["iqr"]) + (((((data["abs_percentile_50"]) + (data["fftr_range_-4000_-3000"]))) * 2.0)))) * 2.0)))) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["mean_change_rate_last_50000"]) - (data["num_peaks_10"]))) - (data["range_-1000_0"]))) - (((data["min_roll_std_100"]) - (data["mean_change_rate_first_50000"]))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((((((-1.0*((((data["percentile_roll_std_5"]) * 2.0))))) - (((1.0) * 2.0)))) - (data["percentile_roll_std_5"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((-1.0*((((((((((data["percentile_95"]) * 2.0)) * 2.0)) * 2.0)) + (data["fftr_skew"])))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((((((-1.0*((data["min_roll_std_10000"])))) - (data["abs_percentile_50"]))) * 2.0)) * 2.0)) * 2.0)) - (((data["ffti_range_0_1000"]) - (data["range_0_1000"]))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["mean_change_rate"]) + (data["percentile_25"]))) * 2.0)) + ((-1.0*(((((data["fftr_max"]) + (data["num_peaks_10"]))/2.0))))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((-1.0*((((((((((data["abs_percentile_70"]) + (data["fftr_range_2000_3000"]))) + (((data["abs_percentile_60"]) + (data["num_peaks_10"]))))) * 2.0)) * 2.0))))) - (data["percentile_roll_std_5"]))) * 2.0)) +

            0.040000*np.tanh((((((((-1.0*(((((data["percentile_roll_std_5"]) + (((((((5.0)) / 2.0)) + (((data["num_peaks_10"]) * 2.0)))/2.0)))/2.0))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((-1.0*((((((((((((((data["ffti_percentile_70"]) + (((data["percentile_roll_std_10"]) + (((data["abs_percentile_70"]) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0))))) * 2.0)) +

            0.040000*np.tanh(((((((((data["percentile_roll_std_70"]) - (((((data["percentile_90"]) - (data["percentile_20"]))) + (data["abs_percentile_70"]))))) * 2.0)) - (((data["percentile_90"]) + (data["min_roll_std_50"]))))) * 2.0)) +

            0.040000*np.tanh(((((((((((data["mean_change_rate"]) * 2.0)) * 2.0)) + (((((((data["mean_change_rate_first_50000"]) - (data["percentile_80"]))) + (data["mean_change_rate_last_10000"]))) - (data["percentile_30"]))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((data["abs_percentile_50"]) + (((((data["range_0_1000"]) + (((((data["range_0_1000"]) - ((((((data["abs_percentile_25"]) + (data["abs_percentile_50"]))/2.0)) * 2.0)))) * 2.0)))) * 2.0)))) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["range_0_1000"]) * 2.0)) + (data["abs_percentile_50"]))) * 2.0)) * 2.0)) - (((data["min_roll_std_500"]) + (((data["percentile_roll_mean_40"]) * 2.0)))))) * 2.0)) +

            0.040000*np.tanh((((((((((-1.0*((((((((((((data["percentile_90"]) * 2.0)) + (data["ffti_percentile_25"]))) * 2.0)) * 2.0)) * 2.0))))) * 2.0)) - (data["percentile_80"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((((((((-1.0*((data["min_roll_std_10000"])))) + ((((((((-1.0*((data["abs_percentile_40"])))) + (data["mean_change_rate"]))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((data["abs_percentile_10"]) - (((data["abs_percentile_50"]) + (((((((data["abs_percentile_70"]) + (data["ffti_range_-3000_-2000"]))) * 2.0)) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((7.85423231124877930)) * (((((data["abs_percentile_50"]) * 2.0)) + (((data["mean_change_rate"]) - (((((data["min_roll_std_10000"]) + (((data["abs_percentile_70"]) - (data["mean_change_rate"]))))) * 2.0)))))))) +

            0.040000*np.tanh((((((((((((((((((((-1.0*((((data["percentile_90"]) * 2.0))))) + (data["fftr_abs_percentile_60"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh((-1.0*(((((((((((((10.67502307891845703)) + (((data["min_roll_std_500"]) * 2.0)))/2.0)) + (((((((data["num_peaks_10"]) * 2.0)) + (data["min_roll_std_1000"]))) * 2.0)))/2.0)) * 2.0)) * 2.0))))) +

            0.040000*np.tanh((((((((((((((((((-1.0*((((data["min_roll_std_10000"]) + (data["abs_percentile_50"])))))) * 2.0)) - (data["ffti_exp_Moving_average_50000_mean"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((data["percentile_roll_std_30"]) + (((((((data["abs_percentile_90"]) - (((data["percentile_90"]) * 2.0)))) * 2.0)) * 2.0)))) - (((data["percentile_90"]) * 2.0)))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["percentile_roll_std_25"]) - (((data["num_peaks_10"]) - (((data["percentile_roll_std_25"]) + (((((((data["percentile_20"]) - (data["percentile_roll_std_20"]))) * 2.0)) * 2.0)))))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh((((((-1.0*((((((((((((((data["abs_percentile_70"]) - (data["percentile_20"]))) * 2.0)) - (data["num_peaks_10"]))) * 2.0)) * 2.0)) - (data["ffti_classic_sta_lta8_mean"])))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["ffti_min_roll_mean_50"]) - (data["autocorrelation_10"]))) - ((((((((((data["min_roll_std_500"]) + (data["num_peaks_100"]))/2.0)) * 2.0)) + (data["abs_percentile_50"]))) * 2.0)))) - (data["min_roll_std_500"]))) +

            0.040000*np.tanh(((((((data["ffti_trend"]) - (((((((data["fftr_range_2000_3000"]) + (data["fftr_range_-2000_-1000"]))) + (((((data["percentile_90"]) * 2.0)) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((data["abs_percentile_70"]) * (((data["percentile_20"]) * 2.0)))) - (data["num_peaks_10"]))) * 2.0)) - (data["abs_percentile_70"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((data["mean_change_rate_last_10000"]) - (data["min_roll_std_500"]))) + (((data["mean_change_rate"]) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) - (data["min_roll_std_50"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((((data["abs_percentile_20"]) - (data["fftr_spkt_welch_density_5"]))) - (data["percentile_90"]))) * 2.0)) - (data["percentile_90"]))) + (data["percentile_roll_std_60"]))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((-1.0) + ((((-1.0*((((((data["abs_percentile_25"]) - (data["mean_change_rate_last_10000"]))) - (((((((-1.0) - (data["num_peaks_10"]))) * 2.0)) * 2.0))))))) * 2.0)))) +

            0.040000*np.tanh(((3.0) * (((((((data["percentile_20"]) + (data["percentile_90"]))) + ((((data["percentile_20"]) + ((-1.0*((-1.0)))))/2.0)))) - (((data["percentile_80"]) * 2.0)))))) +

            0.040000*np.tanh(((((((((data["mean_change_rate"]) * 2.0)) * 2.0)) - (((data["ffti_abs_percentile_40"]) * 2.0)))) * (((((((data["iqr"]) * 2.0)) * 2.0)) * (((data["gmean"]) * 2.0)))))) +

            0.040000*np.tanh(((data["abs_percentile_20"]) + (((((((data["ffti_mean_last_50000"]) - (((data["fftr_percentile_roll_mean_50"]) - (((data["classic_sta_lta5_mean"]) - (((data["ffti_abs_percentile_10"]) + (data["percentile_roll_std_1"]))))))))) * 2.0)) * 2.0)))) +

            0.040000*np.tanh(((data["exp_Moving_average_50000_std"]) + (((data["ffti_percentile_roll_mean_95"]) - (((data["min_roll_std_100"]) + (((((((data["ffti_range_-3000_-2000"]) + (((data["autocorrelation_10"]) - (data["percentile_10"]))))) * 2.0)) * 2.0)))))))) +

            0.040000*np.tanh(((data["mean_change_rate_last_1000"]) - (((((((data["ffti_mean_first_50000"]) + (data["ffti_abs_percentile_5"]))) + (data["num_peaks_50"]))) - (((data["classic_sta_lta3_mean"]) - (((data["abs_percentile_5"]) + (data["percentile_70"]))))))))) +

            0.040000*np.tanh(((((((data["percentile_80"]) * (((data["min_roll_std_10000"]) * (((((((-2.0) - (data["num_peaks_10"]))) - (data["min_roll_std_10000"]))) * 2.0)))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["mean_change_rate_first_1000"]) - (((((((((data["fftr_percentile_roll_std_80"]) + (data["fftr_mean_first_50000"]))) + (data["fftr_percentile_roll_mean_1"]))) + (data["fftr_percentile_roll_mean_1"]))) * 2.0)))) * 2.0)) - (data["min_roll_std_500"]))) +

            0.040000*np.tanh(((((((((((data["abs_percentile_20"]) - (((data["abs_percentile_75"]) + (((((((data["ffti_percentile_roll_std_80"]) * 2.0)) * 2.0)) - (data["percentile_roll_std_70"]))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((data["abs_percentile_50"]) - (data["num_peaks_10"]))) + (((((data["abs_percentile_50"]) - (((data["percentile_roll_std_10"]) + (data["fftr_percentile_roll_std_80"]))))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["autocorrelation_5"]) + (((((((data["fftr_av_change_abs_roll_std_10"]) - (((((((data["abs_percentile_75"]) + (data["ffti_percentile_roll_std_75"]))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((data["ffti_mean_last_50000"]) - (((data["fftr_percentile_roll_mean_75"]) - (-1.0))))) - (data["min_roll_std_100"]))) + (((data["mean_change_rate_first_1000"]) + (data["skew"]))))) - (data["fftr_percentile_roll_mean_75"]))) +

            0.040000*np.tanh(((((((((((data["percentile_roll_std_10"]) + ((((-1.0*((data["min_roll_std_10000"])))) * (((data["percentile_roll_std_10"]) * (data["percentile_roll_std_10"]))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((data["ffti_percentile_75"]) - (((((data["range_-1000_0"]) * (((data["percentile_roll_std_5"]) * (data["percentile_80"]))))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((data["percentile_10"]) - (((data["classic_sta_lta7_mean"]) * (((((data["fftr_range_2000_3000"]) + (data["fftr_av_change_abs_roll_std_50"]))) - (((((data["exp_Moving_average_50000_std"]) - (data["fftr_percentile_40"]))) * 2.0)))))))) * 2.0)) +

            0.040000*np.tanh((((((-1.0*((((((((data["fftr_percentile_roll_mean_10"]) * 2.0)) * 2.0)) * (((data["fftr_min_roll_mean_10000"]) - (data["percentile_roll_std_75"])))))))) - (((((data["ffti_percentile_roll_std_80"]) * 2.0)) * 2.0)))) * 2.0)) +

            0.040000*np.tanh((((((((((data["ffti_percentile_70"]) + ((((((data["autocorrelation_100"]) + (((data["num_peaks_10"]) * (data["num_peaks_10"]))))/2.0)) * 2.0)))/2.0)) * (data["fftr_percentile_20"]))) * 2.0)) - (data["fftr_percentile_20"]))) +

            0.040000*np.tanh(((((((((data["num_peaks_10"]) - (((data["percentile_roll_std_10"]) * ((((((data["fftr_percentile_roll_std_75"]) + (((data["min_roll_std_10000"]) * (data["num_peaks_10"]))))/2.0)) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((((((((data["fftr_percentile_30"]) * (data["fftr_abs_percentile_40"]))) - (((data["mean_change_rate_first_50000"]) * (data["ffti_max_roll_mean_10000"]))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) + (data["classic_sta_lta6_mean"]))) +

            0.040000*np.tanh(((((((((data["autocorrelation_5"]) - (((((((data["fftr_percentile_roll_std_80"]) * 2.0)) * 2.0)) + (data["fftr_percentile_roll_std_80"]))))) - (((data["autocorrelation_5"]) * (data["abs_mean"]))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((((data["ffti_min_roll_std_10000"]) - (((data["percentile_40"]) * (data["iqr"]))))) - (((data["ave_roll_std_100"]) - (((data["fftr_percentile_75"]) * (data["fftr_exp_Moving_average_3000_mean"]))))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((data["ffti_mean_last_50000"]) + (((data["abs_percentile_70"]) - (((((((((data["ffti_exp_Moving_average_50000_std"]) * 2.0)) * (data["ffti_range_p4000_pinf"]))) * 2.0)) + ((((data["min_roll_std_50"]) + (data["fftr_autocorrelation_50"]))/2.0)))))))) +

            0.040000*np.tanh(((data["ffti_min_first_1000"]) - (((((((data["abs_percentile_25"]) + (((((data["fftr_ave_roll_mean_10000"]) - ((-1.0*((data["ffti_percentile_roll_mean_50"])))))) * 2.0)))) + (data["fftr_time_rev_asym_stat_5"]))) + (data["min_roll_std_50"]))))) +

            0.040000*np.tanh(((((data["percentile_roll_std_60"]) * 2.0)) - (((((data["ffti_range_2000_3000"]) + (data["fftr_abs_percentile_70"]))) * (((((((data["num_peaks_10"]) * (data["num_peaks_10"]))) + (data["fftr_abs_percentile_70"]))) * 2.0)))))) +

            0.040000*np.tanh((-1.0*((((((((((((data["fftr_abs_max_roll_mean_100"]) * 2.0)) + (((data["mean_last_50000"]) + (((data["ffti_av_change_abs_roll_std_10000"]) - (data["ffti_min_last_1000"]))))))) + (data["ffti_autocorrelation_50"]))) * 2.0)) * 2.0))))) +

            0.040000*np.tanh(((((data["abs_percentile_70"]) + (((data["fftr_Hann_window_mean_50"]) + (((((((((data["percentile_60"]) * (((data["percentile_20"]) * (data["abs_percentile_40"]))))) * 2.0)) * 2.0)) * 2.0)))))) * 2.0)) +

            0.040000*np.tanh(((((((data["fftr_min_roll_std_500"]) - (data["fftr_kstat_1"]))) + (data["av_change_abs_roll_mean_1000"]))) + (((data["ffti_max_to_min"]) - (((((data["exp_Moving_average_50000_std"]) * (((data["percentile_roll_std_80"]) * 2.0)))) * 2.0)))))) +

            0.040000*np.tanh(((((data["fftr_percentile_roll_mean_70"]) * (((data["ffti_abs_percentile_1"]) + (data["fftr_range_3000_4000"]))))) - (((data["fftr_range_p4000_pinf"]) * (((data["ffti_abs_percentile_1"]) + (((((data["fftr_min_roll_std_500"]) * 2.0)) * 2.0)))))))) +

            0.040000*np.tanh(((((data["fftr_abs_percentile_5"]) * (((((((((((((data["fftr_percentile_roll_mean_80"]) - (data["ffti_iqr"]))) + (data["percentile_roll_mean_1"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) + (data["percentile_20"]))) +

            0.040000*np.tanh(((((((((data["percentile_roll_std_90"]) - ((-1.0*((((((data["min_roll_std_10000"]) * (data["ffti_min_last_10000"]))) * (((data["min_roll_std_100"]) + (data["abs_percentile_90"])))))))))) * 2.0)) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["mean_change_rate_first_1000"]) - (data["classic_sta_lta7_mean"]))) + ((-1.0*((((((data["fftr_mean_first_50000"]) + (((data["fftr_percentile_roll_mean_70"]) - (data["ffti_skew"]))))) + (data["fftr_percentile_roll_std_80"])))))))) * 2.0)) +

            0.040000*np.tanh(((data["percentile_75"]) - (((((data["ffti_av_change_abs_roll_std_50"]) - (((data["fftr_percentile_roll_std_40"]) - (((((((data["fftr_percentile_roll_std_75"]) - (data["min_roll_mean_500"]))) * 2.0)) * 2.0)))))) * 2.0)))) +

            0.040000*np.tanh(((((((data["fftr_min_roll_std_500"]) + (((((data["classic_sta_lta6_mean"]) + (((((data["fftr_min_first_10000"]) * 2.0)) * 2.0)))) + ((((data["ffti_min_roll_std_10"]) + (data["ffti_std_roll_mean_10000"]))/2.0)))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((data["percentile_60"]) + (((((data["fftr_percentile_roll_std_50"]) - (((data["percentile_60"]) * (((data["fftr_percentile_60"]) + (((data["fftr_av_change_abs_roll_std_10000"]) + (data["ffti_autocorrelation_10000"]))))))))) * 2.0)))) * 2.0)) +

            0.040000*np.tanh(((data["fftr_num_peaks_50"]) + (((data["ffti_min_roll_std_10000"]) + (((((data["percentile_20"]) - (data["av_change_abs_roll_std_500"]))) + (((data["ffti_percentile_40"]) + (((data["fftr_hmean"]) - (data["percentile_70"]))))))))))) +

            0.040000*np.tanh(((data["autocorrelation_50"]) + (((data["percentile_70"]) * ((((((((((data["percentile_75"]) * (data["range_0_1000"]))) * 2.0)) + (((data["percentile_70"]) * (data["percentile_20"]))))/2.0)) * 2.0)))))) +

            0.040000*np.tanh(((((data["percentile_60"]) * 2.0)) * (((data["mean_change_rate_first_10000"]) - (((((data["fftr_mean_last_50000"]) + (((data["min_last_10000"]) + (((data["percentile_roll_std_25"]) * 2.0)))))) * (data["ffti_abs_max_roll_mean_10"]))))))) +

            0.040000*np.tanh(((((((((data["mean_change_rate_last_10000"]) * (data["ffti_autocorrelation_5000"]))) * 2.0)) + (data["ffti_percentile_roll_mean_95"]))) + (((data["fftr_mean_first_50000"]) * (((((data["fftr_range_2000_3000"]) - (data["fftr_mean_first_50000"]))) * 2.0)))))) +

            0.040000*np.tanh((((((-1.0*((((((((data["ffti_abs_percentile_60"]) * (data["ffti_percentile_roll_std_30"]))) + (((((data["ffti_percentile_roll_std_20"]) * (data["fftr_min_roll_std_50"]))) + (data["fftr_spkt_welch_density_5"]))))) * 2.0))))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((((((data["ffti_mean_last_50000"]) + (data["fftr_time_rev_asym_stat_10"]))) - (((data["min_roll_std_50"]) - (((((data["autocorrelation_500"]) + (data["mean_change_rate_last_1000"]))) + (data["ffti_max_to_min"]))))))) * 2.0)) +

            0.040000*np.tanh(((data["ffti_classic_sta_lta5_mean"]) - (((data["ffti_std_first_1000"]) - (((((data["ffti_iqr"]) - (((data["ffti_range_2000_3000"]) * ((-1.0*((data["percentile_roll_std_10"])))))))) * ((-1.0*((data["ffti_range_2000_3000"])))))))))) +

            0.040000*np.tanh(((((data["abs_percentile_70"]) + (((data["classic_sta_lta5_mean"]) * (((((((data["ffti_iqr"]) * (((data["classic_sta_lta5_mean"]) - (data["num_peaks_10"]))))) * 2.0)) * 2.0)))))) - (data["autocorrelation_10000"]))) +

            0.040000*np.tanh(((data["ffti_range_minf_m4000"]) * (((data["fftr_percentile_roll_mean_5"]) - (((((((((data["fftr_mean_first_50000"]) * (((data["ffti_kurt"]) + (data["fftr_percentile_60"]))))) - (data["fftr_percentile_roll_mean_5"]))) * 2.0)) * 2.0)))))) +

            0.040000*np.tanh(((((data["ffti_range_3000_4000"]) * 2.0)) * (((data["ffti_percentile_roll_mean_80"]) - (((((((data["autocorrelation_100"]) + (data["ffti_percentile_roll_std_50"]))) + (data["autocorrelation_5"]))) + (((data["ffti_percentile_roll_std_25"]) * 2.0)))))))) +

            0.040000*np.tanh(((((data["ffti_abs_percentile_1"]) + (((data["min_last_50000"]) + (((data["autocorrelation_50"]) * (((data["mean_last_1000"]) - (data["ffti_percentile_roll_std_40"]))))))))) + (((data["fftr_range_2000_3000"]) * (data["fftr_ave_roll_mean_1000"]))))) +

            0.040000*np.tanh(((((((data["ffti_percentile_roll_std_40"]) + (data["skew"]))) - (((data["fftr_percentile_roll_std_5"]) - (((data["ffti_percentile_roll_std_50"]) - (data["fftr_percentile_roll_mean_20"]))))))) - ((((data["ffti_abs_max_roll_mean_10"]) + (data["ffti_av_change_abs_roll_std_10"]))/2.0)))) +

            0.040000*np.tanh((-1.0*((((data["percentile_roll_std_80"]) * (((data["classic_sta_lta2_mean"]) + (((data["fftr_percentile_roll_std_30"]) - (((data["mean_change_rate_first_1000"]) * (((((data["ffti_range_-4000_-3000"]) * 2.0)) + (data["percentile_roll_std_30"])))))))))))))) +

            0.040000*np.tanh(((((data["ffti_std_roll_mean_10000"]) * (((data["min_roll_std_50"]) - (data["fftr_max_roll_mean_100"]))))) - (((data["fftr_percentile_roll_std_20"]) + (((data["ffti_kstat_1"]) + (((data["fftr_classic_sta_lta5_mean"]) * (data["fftr_autocorrelation_10"]))))))))) +

            0.040000*np.tanh(((((((data["ffti_range_-3000_-2000"]) + (((data["ffti_autocorrelation_50"]) * (data["ffti_num_peaks_50"]))))) - (((data["ffti_abs_percentile_60"]) * (data["ffti_range_-3000_-2000"]))))) - (((data["ffti_range_-3000_-2000"]) * (data["ffti_range_-3000_-2000"]))))) +

            0.040000*np.tanh(((((((((((((data["fftr_mean_change_abs"]) + (((((data["ffti_percentile_roll_mean_50"]) + (data["fftr_mean_change_abs"]))) * (data["fftr_av_change_abs_roll_std_10"]))))) * 2.0)) * 2.0)) * 2.0)) - (data["ffti_min_roll_std_50"]))) * 2.0)) +

            0.040000*np.tanh(((((((data["abs_percentile_70"]) - (((data["ffti_percentile_roll_mean_5"]) - (((data["mean_change_rate_first_50000"]) - (data["fftr_max_roll_std_100"]))))))) * 2.0)) - (((data["ffti_percentile_roll_mean_5"]) - (data["mean_change_rate_first_50000"]))))) +

            0.040000*np.tanh(((((data["min_last_10000"]) + (((data["ffti_percentile_roll_std_1"]) + (((((data["fftr_range_-3000_-2000"]) * (data["fftr_percentile_roll_mean_1"]))) * 2.0)))))) + (((((data["fftr_range_-3000_-2000"]) * (data["ffti_percentile_roll_std_1"]))) * 2.0)))) +

            0.039984*np.tanh(((data["ffti_min_first_1000"]) + (((((data["fftr_av_change_abs_roll_std_1000"]) * (((((data["ffti_med"]) + (data["fftr_time_rev_asym_stat_100"]))) + (data["fftr_time_rev_asym_stat_100"]))))) + (((data["fftr_abs_percentile_40"]) * (data["fftr_num_peaks_20"]))))))) +

            0.040000*np.tanh(((((data["ffti_spkt_welch_density_10"]) - (((data["fftr_time_rev_asym_stat_5"]) + (((data["ffti_min_roll_std_10000"]) - (((data["ffti_classic_sta_lta5_mean"]) * (data["fftr_percentile_roll_mean_70"]))))))))) - (((data["ffti_percentile_roll_mean_50"]) + (data["ffti_percentile_roll_std_10"]))))) +

            0.035107*np.tanh(((((((((data["fftr_autocorrelation_100"]) * (data["fftr_autocorrelation_100"]))) + (((((data["ffti_min_roll_std_50"]) * ((-1.0*(((((data["fftr_autocorrelation_100"]) + (data["fftr_max_last_1000"]))/2.0))))))) * 2.0)))) * 2.0)) * 2.0)) +

            0.039984*np.tanh(((((data["abs_percentile_20"]) - (data["ffti_min_roll_std_50"]))) + (((data["classic_sta_lta3_mean"]) + ((((((-1.0*((((data["fftr_num_peaks_20"]) * 2.0))))) * (data["fftr_range_0_1000"]))) - (data["fftr_num_peaks_20"]))))))) +

            0.040000*np.tanh(((data["classic_sta_lta4_mean"]) * (((((((((data["ffti_num_peaks_10"]) + (data["fftr_hmean"]))) + (data["ffti_percentile_roll_mean_10"]))) - (((data["max_first_10000"]) - (data["ffti_num_peaks_10"]))))) - (data["autocorrelation_5"]))))) +

            0.040000*np.tanh(((data["fftr_ave_roll_mean_100"]) - (((data["num_peaks_10"]) * (((data["fftr_ave_roll_mean_100"]) * (((((data["ffti_classic_sta_lta2_mean"]) * 2.0)) * 2.0)))))))) +

            0.040000*np.tanh(((((((data["ffti_autocorrelation_10000"]) * (((data["ffti_abs_max_roll_mean_10"]) - ((((data["ffti_max_to_min"]) + (((data["ffti_percentile_roll_mean_90"]) - (((data["ffti_abs_max_roll_mean_10"]) * (data["ffti_exp_Moving_average_300_mean"]))))))/2.0)))))) * 2.0)) * 2.0)) +

            0.039984*np.tanh(((data["min_roll_std_100"]) * ((((-1.0*((data["ffti_percentile_roll_std_50"])))) + (((data["fftr_time_rev_asym_stat_1"]) - (((((data["exp_Moving_average_30000_std"]) - (data["ffti_autocorrelation_100"]))) - ((-1.0*((data["skew"])))))))))))) +

            0.040000*np.tanh(((((data["ffti_skew"]) * (((data["min_roll_std_50"]) - (data["fftr_min_roll_std_10000"]))))) + (((data["ffti_percentile_roll_std_40"]) + (((data["fftr_percentile_roll_mean_70"]) * (data["fftr_time_rev_asym_stat_1"]))))))) +

            0.039984*np.tanh(((((data["fftr_percentile_roll_mean_1"]) + (data["fftr_percentile_roll_mean_1"]))) * ((-1.0*((((((((data["fftr_percentile_roll_mean_1"]) + (data["fftr_percentile_roll_mean_1"]))) + (((data["ffti_max"]) + (data["ffti_av_change_abs_roll_mean_10000"]))))) * 2.0))))))) +

            0.040000*np.tanh((-1.0*((((data["av_change_abs_roll_std_500"]) * (((((((((((data["ffti_kstat_1"]) + (data["min_roll_std_1000"]))) + (data["ffti_av_change_abs_roll_std_10000"]))) * 2.0)) + (data["fftr_min_roll_std_50"]))) * 2.0))))))) +

            0.040000*np.tanh(((((((((data["fftr_percentile_roll_std_20"]) + (((data["ffti_av_change_abs_roll_std_50"]) + (data["percentile_50"]))))) + (data["autocorrelation_1000"]))) * (((data["percentile_10"]) * (data["fftr_iqr"]))))) * 2.0)) +

            0.040000*np.tanh(((data["min_roll_std_500"]) * (((data["ffti_percentile_roll_std_1"]) + (((((((((((data["mean_change_rate_last_1000"]) * (data["ffti_av_change_abs_roll_std_10000"]))) * 2.0)) + (data["ffti_av_change_abs_roll_std_10000"]))) * 2.0)) - (data["std_last_50000"]))))))) +

            0.039984*np.tanh(((data["kstat_1"]) * (((((((((data["fftr_min_roll_std_500"]) + ((((((data["classic_sta_lta6_mean"]) - (data["iqr"]))) + (data["autocorrelation_50"]))/2.0)))) * 2.0)) * 2.0)) + (data["ffti_max_to_min_diff"]))))) +

            0.040000*np.tanh(((((((((data["fftr_min_roll_std_10000"]) * (((data["fftr_min_roll_mean_500"]) - (data["spkt_welch_density_1"]))))) * 2.0)) * 2.0)) + (((data["fftr_abs_percentile_5"]) + (((data["classic_sta_lta4_mean"]) * (data["ffti_exp_Moving_average_50000_mean"]))))))) +

            0.040000*np.tanh(((data["fftr_autocorrelation_1000"]) * (((data["ffti_av_change_abs_roll_mean_500"]) + (((((((data["fftr_autocorrelation_1000"]) * (data["ffti_kstat_1"]))) + (((data["abs_max_roll_std_100"]) + (data["ffti_av_change_abs_roll_mean_500"]))))) * (data["num_peaks_10"]))))))) +

            0.039984*np.tanh(((data["min_roll_std_50"]) * (((data["percentile_roll_std_25"]) + (((((data["ffti_abs_max_roll_mean_50"]) + (((data["abs_mean"]) * (((data["mean_change_rate_last_10000"]) - (data["percentile_roll_std_25"]))))))) - (data["min_roll_std_50"]))))))) +

            0.039984*np.tanh((((((data["fftr_min_roll_std_500"]) - (data["av_change_abs_roll_mean_50"]))) + (((data["classic_sta_lta6_mean"]) + (((data["ffti_med"]) - (((((data["av_change_abs_roll_std_500"]) * (data["ffti_percentile_roll_mean_40"]))) * 2.0)))))))/2.0)) +

            0.040000*np.tanh(((((((((((data["exp_Moving_average_50000_std"]) * (data["ffti_percentile_roll_mean_50"]))) * 2.0)) + (((data["ffti_percentile_roll_mean_50"]) * (data["ffti_percentile_roll_mean_50"]))))) * 2.0)) + (((data["fftr_ave_roll_mean_1000"]) * (data["fftr_autocorrelation_5"]))))) +

            0.039875*np.tanh(((data["classic_sta_lta8_mean"]) * (((((data["autocorrelation_1000"]) + (((data["fftr_trend"]) + (((((((data["ffti_av_change_abs_roll_std_500"]) * 2.0)) + (data["fftr_percentile_roll_mean_25"]))) * 2.0)))))) + (data["fftr_percentile_roll_mean_25"]))))) +

            0.040000*np.tanh(((data["ffti_percentile_roll_mean_95"]) * (((((((data["av_change_abs_roll_mean_50"]) * 2.0)) - ((-1.0*((((((((data["av_change_abs_roll_mean_50"]) - (data["ffti_min_roll_std_500"]))) * 2.0)) * 2.0))))))) - (data["mean_change_rate_first_10000"]))))) +

            0.040000*np.tanh((((((data["autocorrelation_1000"]) + ((((data["autocorrelation_50"]) + (data["mean_change_rate_last_10000"]))/2.0)))/2.0)) - ((-1.0*((((data["classic_sta_lta2_mean"]) * (((data["fftr_time_rev_asym_stat_1"]) - (data["ffti_av_change_abs_roll_mean_10000"])))))))))) +

            0.040000*np.tanh(((data["ffti_min_last_10000"]) - (((data["ffti_av_change_abs_roll_std_10"]) * (((((data["autocorrelation_50"]) + (data["ffti_spkt_welch_density_5"]))) + (((((data["ffti_spkt_welch_density_5"]) + (data["ffti_min_last_10000"]))) + (data["av_change_abs_roll_std_500"]))))))))) +

            0.039984*np.tanh(((data["ffti_max_to_min"]) * (((((data["ave_roll_mean_100"]) + (data["ffti_med"]))) + (((((((data["ave_roll_mean_100"]) + (data["ffti_percentile_50"]))) - (data["mean_change_rate_last_1000"]))) + (data["fftr_min_last_1000"]))))))) +

            0.040000*np.tanh(((((data["min_roll_std_100"]) - (data["med"]))) * (((((data["percentile_90"]) - (((data["min_roll_std_500"]) * (data["ffti_max_roll_mean_50"]))))) - (((data["ffti_kurt"]) - (data["min_roll_std_100"]))))))) +

            0.039984*np.tanh(((data["ffti_percentile_roll_mean_60"]) * (((data["fftr_abs_percentile_1"]) + (((((data["autocorrelation_100"]) + (((data["ffti_mean_last_1000"]) - (data["ffti_percentile_roll_mean_60"]))))) * 2.0)))))) +

            0.040000*np.tanh(((data["max_first_10000"]) * (((((data["max_first_10000"]) + (((((data["fftr_spkt_welch_density_10"]) - (((data["ffti_spkt_welch_density_5"]) + (((data["skew"]) + (data["ffti_autocorrelation_5"]))))))) * 2.0)))) * 2.0)))) +

            0.040000*np.tanh(((data["fftr_percentile_roll_std_60"]) + (((data["fftr_mean_first_1000"]) * (((data["percentile_30"]) + (((data["fftr_mean_first_1000"]) * (((data["fftr_percentile_70"]) * (((data["ffti_range_-1000_0"]) - (data["fftr_percentile_70"]))))))))))))) +

            0.040000*np.tanh(((((data["fftr_time_rev_asym_stat_10"]) * (data["ffti_percentile_50"]))) + (((((data["exp_Moving_average_50000_std"]) * (data["autocorrelation_500"]))) + (((data["fftr_time_rev_asym_stat_5"]) * (((data["fftr_time_rev_asym_stat_10"]) + (data["min_first_1000"]))))))))) +

            0.040000*np.tanh((((((((data["max_roll_mean_10"]) + (data["fftr_min_roll_std_50"]))/2.0)) + (((data["classic_sta_lta4_mean"]) * (data["fftr_hmean"]))))) + (((data["percentile_roll_std_80"]) * (data["percentile_25"]))))) +

            0.040000*np.tanh(((((data["fftr_mean_last_1000"]) * (((data["fftr_mean_last_1000"]) + ((((-1.0*((((data["ffti_percentile_roll_std_50"]) + (data["fftr_mean_last_50000"])))))) * 2.0)))))) - (((data["ffti_autocorrelation_5000"]) * (data["exp_Moving_average_50000_std"]))))) +

            0.040000*np.tanh(((((data["exp_Moving_average_50000_std"]) / 2.0)) + (((data["ffti_percentile_roll_mean_99"]) + (((data["mean_change_abs"]) * ((-1.0*((((data["exp_Moving_average_50000_std"]) + ((((data["av_change_rate_roll_std_1000"]) + (data["fftr_num_peaks_10"]))/2.0))))))))))))) +

            0.039984*np.tanh((((data["ffti_av_change_abs_roll_std_1000"]) + ((((((((data["min_last_1000"]) + (((data["min_last_1000"]) / 2.0)))/2.0)) - (data["ffti_max_first_1000"]))) - ((((data["fftr_spkt_welch_density_5"]) + (data["fftr_abs_max_roll_mean_100"]))/2.0)))))/2.0)) +

            0.039984*np.tanh(((data["ffti_autocorrelation_1000"]) * (((data["ffti_std_last_10000"]) - ((((data["num_peaks_10"]) + ((((data["ffti_ave_roll_mean_100"]) + (((data["ffti_sum"]) * (((((data["ffti_percentile_roll_mean_10"]) * 2.0)) * 2.0)))))/2.0)))/2.0)))))) +

            0.040000*np.tanh(((data["autocorrelation_500"]) * (((data["percentile_70"]) + (((data["exp_Moving_average_50000_std"]) + (((data["percentile_80"]) - (((data["ffti_min_roll_std_100"]) + (((data["ffti_mean_change_rate"]) + (data["fftr_min_roll_std_50"]))))))))))))) +

            0.039969*np.tanh(((data["ffti_min_roll_mean_100"]) * (((((((data["percentile_roll_std_40"]) * (data["min_roll_std_1000"]))) + (data["ffti_av_change_abs_roll_std_10"]))) - (((((data["fftr_mean"]) + (data["ffti_kurt"]))) + (data["ffti_percentile_roll_mean_95"]))))))) +

            0.040000*np.tanh(((((((((data["classic_sta_lta6_mean"]) - (data["max_last_10000"]))) - ((((data["min_first_10000"]) + (((data["ffti_min_roll_std_100"]) + (data["max_last_10000"]))))/2.0)))) + (data["classic_sta_lta4_mean"]))) - (data["min_first_10000"]))) +

            0.039891*np.tanh((((((data["mean_change_rate_last_10000"]) - (data["percentile_50"]))) + (((-1.0) + (((((-1.0) + (((data["mean_change_rate_last_10000"]) * (data["mean_change_rate_last_10000"]))))) + (data["ffti_mean_change_abs"]))))))/2.0)) +

            0.039984*np.tanh((((((data["ffti_mean_first_10000"]) * (data["fftr_time_rev_asym_stat_1"]))) + ((((data["min_roll_std_10"]) + (((((data["av_change_abs_roll_mean_500"]) + (((data["fftr_hmean"]) * (data["skew"]))))) + (data["fftr_hmean"]))))/2.0)))/2.0)) +

            0.040000*np.tanh(((data["fftr_range_2000_3000"]) * (((data["percentile_10"]) - ((((((((data["num_peaks_10"]) + (data["percentile_roll_mean_99"]))/2.0)) * (((data["ffti_av_change_abs_roll_std_10000"]) - (data["percentile_10"]))))) * 2.0)))))) +

            0.040000*np.tanh(((data["std_last_1000"]) * (((data["autocorrelation_10000"]) - (((data["fftr_hmean"]) * ((((((data["fftr_classic_sta_lta6_mean"]) * 2.0)) + (((((data["ffti_av_change_abs_roll_mean_10000"]) * 2.0)) * 2.0)))/2.0)))))))) +

            0.039984*np.tanh(((data["ffti_abs_percentile_20"]) * (((((data["fftr_ave_roll_mean_1000"]) * ((-1.0*((((data["ffti_classic_sta_lta6_mean"]) * 2.0))))))) + (((((data["fftr_percentile_roll_std_50"]) + (data["fftr_min_last_1000"]))) + (data["fftr_min_last_1000"]))))))) +

            0.040000*np.tanh((((((((((data["fftr_ave_roll_mean_1000"]) - (data["fftr_abs_max_roll_mean_10000"]))) + (data["percentile_90"]))) - (data["fftr_max_last_50000"]))) + (((data["ffti_autocorrelation_1000"]) * (((data["ffti_std_roll_std_100"]) - (data["ffti_range_-3000_-2000"]))))))/2.0)) +

            0.039953*np.tanh(((((((data["ffti_min_roll_std_10"]) + (data["min_roll_std_500"]))) + (((data["min_roll_mean_1000"]) + (data["fftr_range_2000_3000"]))))) * (((data["ffti_abs_max_roll_mean_500"]) - (((data["fftr_abs_percentile_20"]) / 2.0)))))) +

            0.040000*np.tanh(((data["percentile_roll_std_70"]) * (((((data["ffti_av_change_abs_roll_std_500"]) + (((data["ffti_av_change_abs_roll_std_10000"]) + (((data["exp_Moving_std_3000_mean"]) * (((data["fftr_percentile_roll_std_10"]) - (data["percentile_roll_mean_90"]))))))))) + (data["ffti_sum"]))))) +

            0.039062*np.tanh(((((data["mean_change_rate_first_10000"]) / 2.0)) - (((data["fftr_mean_change_rate_last_1000"]) - (((data["fftr_range_-3000_-2000"]) - ((((((data["fftr_percentile_roll_std_10"]) + (data["ffti_av_change_abs_roll_std_500"]))/2.0)) + (data["fftr_mean_change_rate_last_1000"]))))))))) +

            0.040000*np.tanh((-1.0*((((data["fftr_mean_last_1000"]) * (((((((data["ffti_abs_max"]) + (data["ffti_percentile_roll_mean_75"]))) - (data["ffti_av_change_abs_roll_std_100"]))) + (((data["abs_percentile_20"]) - (data["ffti_mean"])))))))))) +

            0.038578*np.tanh(((((((((data["fftr_range_p4000_pinf"]) * (((data["mean_change_abs"]) * (data["classic_sta_lta8_mean"]))))) * 2.0)) * 2.0)) + ((((data["fftr_max_roll_mean_500"]) + ((((data["fftr_max_roll_mean_500"]) + (data["autocorrelation_500"]))/2.0)))/2.0)))) +

            0.040000*np.tanh(((data["abs_percentile_70"]) * ((((data["autocorrelation_5000"]) + (((((data["fftr_percentile_roll_mean_90"]) + ((((((data["abs_percentile_20"]) + (data["fftr_percentile_roll_mean_80"]))/2.0)) + (data["fftr_percentile_roll_mean_90"]))))) * (data["ffti_ave_roll_mean_50"]))))/2.0)))) +

            0.040000*np.tanh(((((data["fftr_percentile_roll_std_20"]) * (((((data["ffti_autocorrelation_500"]) - (data["fftr_percentile_roll_mean_70"]))) - (data["num_peaks_100"]))))) - (((data["fftr_percentile_50"]) * (data["ffti_percentile_roll_mean_95"]))))) +

            0.039969*np.tanh(((data["fftr_hmean"]) * ((((((data["abs_std"]) + (((data["fftr_mean_first_50000"]) + (((data["fftr_exp_Moving_average_300_mean"]) - (data["fftr_time_rev_asym_stat_50"]))))))) + (((data["ffti_percentile_roll_mean_75"]) + (data["autocorrelation_10000"]))))/2.0)))) +

            0.039984*np.tanh(((((((data["fftr_time_rev_asym_stat_1"]) - (data["ffti_percentile_roll_mean_40"]))) - (data["fftr_mean"]))) * ((((data["ffti_min_roll_std_10"]) + (((((data["ffti_hmean"]) * (data["mean_last_10000"]))) + (data["fftr_spkt_welch_density_5"]))))/2.0)))) +

            0.039984*np.tanh(((data["ffti_av_change_abs_roll_mean_10000"]) * (((((data["fftr_abs_percentile_5"]) * 2.0)) - (((((data["max_last_50000"]) + (data["exp_Moving_average_50000_mean"]))) + (((data["exp_Moving_average_50000_std"]) - (((data["fftr_abs_percentile_5"]) * 2.0)))))))))) +

            0.040000*np.tanh(((data["fftr_hmean"]) * ((((data["ffti_percentile_roll_std_20"]) + ((((((-1.0*((data["fftr_num_peaks_50"])))) + (data["ffti_av_change_abs_roll_std_100"]))) + (((data["ffti_autocorrelation_1000"]) + (data["ffti_max_to_min_diff"]))))))/2.0)))) +

            0.039969*np.tanh(((data["ffti_Hann_window_mean_50"]) * (((data["percentile_roll_std_80"]) + (((data["ffti_percentile_roll_mean_50"]) + (((((data["percentile_roll_std_80"]) - (((data["ffti_sum"]) * 2.0)))) + (data["fftr_av_change_abs_roll_mean_10000"]))))))))) +

            0.039984*np.tanh(((((data["percentile_60"]) * (((data["ffti_mean_change_rate"]) * 2.0)))) - (((((data["fftr_mean_first_50000"]) * (((((data["ffti_percentile_roll_std_10"]) * 2.0)) * 2.0)))) * 2.0)))) +

            0.040000*np.tanh(((data["fftr_mean"]) * (((((((data["mean_change_rate"]) * (data["ffti_autocorrelation_10000"]))) + (data["percentile_roll_std_75"]))) + ((((((data["mean_change_rate_first_1000"]) + (data["percentile_roll_std_60"]))/2.0)) * (data["fftr_min_roll_mean_1000"]))))))) +

            0.040000*np.tanh(((data["ffti_percentile_roll_mean_60"]) * (((((data["abs_percentile_25"]) + (((data["ffti_skew"]) * (data["percentile_roll_mean_25"]))))) - (((data["fftr_max_first_10000"]) * (((data["ffti_max_to_min"]) + (data["ffti_abs_percentile_60"]))))))))) +

            0.040000*np.tanh((((((((data["autocorrelation_100"]) * (data["ffti_min_roll_std_1000"]))) + (((data["av_change_abs_roll_std_100"]) * (data["ffti_exp_Moving_average_50000_mean"]))))) + (((data["autocorrelation_10000"]) * (((data["ffti_exp_Moving_average_50000_mean"]) * (data["autocorrelation_100"]))))))/2.0)) +

            0.040000*np.tanh(((((((((data["ffti_sum"]) * (data["ffti_kstat_1"]))) * 2.0)) - (((data["num_peaks_50"]) / 2.0)))) + (((((data["num_peaks_50"]) * (data["ffti_sum"]))) - (data["ffti_kstat_1"]))))) +

            0.039984*np.tanh((((data["ffti_min_roll_mean_10"]) + (((((data["fftr_trend"]) * ((((((((data["ffti_min_roll_std_50"]) + (data["min_roll_std_1000"]))) + (data["ffti_min_roll_std_50"]))/2.0)) - (((data["ffti_min_roll_std_10"]) / 2.0)))))) * 2.0)))/2.0)) +

            0.040000*np.tanh(((((data["ffti_min_roll_std_500"]) * (((data["ffti_mean_first_50000"]) + (((((data["min_roll_std_10"]) * (((data["ffti_mean_first_50000"]) * 2.0)))) + (((data["min_roll_std_10"]) * (data["ffti_exp_Moving_average_50000_std"]))))))))) * 2.0)) +

            0.039984*np.tanh(((((data["ffti_min_roll_std_10"]) * (data["max_roll_mean_100"]))) + (((data["mean_change_abs"]) * (((((data["fftr_percentile_roll_std_40"]) - (((data["ffti_min_roll_std_10"]) * (data["abs_percentile_20"]))))) + (data["fftr_percentile_roll_std_40"]))))))) +

            0.040000*np.tanh(((data["ffti_mean_last_50000"]) * (((((((data["exp_Moving_average_50000_std"]) * (((data["ffti_autocorrelation_1000"]) + (data["ffti_percentile_roll_std_25"]))))) - ((((data["fftr_percentile_roll_std_10"]) + (data["percentile_roll_std_25"]))/2.0)))) - (data["ffti_percentile_roll_std_20"]))))) +

            0.040000*np.tanh(((data["percentile_60"]) * (((data["percentile_60"]) * (((data["percentile_60"]) * ((((data["ffti_av_change_abs_roll_std_10"]) + (((data["num_peaks_20"]) - (((data["fftr_time_rev_asym_stat_10"]) * (data["abs_percentile_20"]))))))/2.0)))))))) +

            0.039969*np.tanh(((data["ffti_max_roll_mean_1000"]) * (((((((((((data["fftr_min_first_1000"]) * (data["abs_percentile_70"]))) * 2.0)) + (data["fftr_min_first_1000"]))) + ((((-1.0*((data["ffti_percentile_roll_std_50"])))) * 2.0)))) * 2.0)))) +

            0.040000*np.tanh(((data["ffti_percentile_roll_std_1"]) * (((((((data["classic_sta_lta8_mean"]) + (data["fftr_percentile_roll_std_40"]))) + (((((data["fftr_percentile_roll_std_40"]) - (data["ffti_num_peaks_10"]))) - (data["ffti_percentile_roll_std_1"]))))) - (data["abs_percentile_20"]))))) +

            0.040000*np.tanh(((((((data["ffti_av_change_abs_roll_std_500"]) * ((((((data["min_roll_std_500"]) + (((data["ffti_autocorrelation_100"]) * (data["ffti_av_change_abs_roll_mean_10000"]))))/2.0)) * 2.0)))) * 2.0)) * 2.0)) +

            0.031856*np.tanh(((((data["max_to_min"]) * (((data["ffti_Hann_window_mean_15000"]) + ((((data["fftr_min_roll_std_10"]) + (data["max_to_min"]))/2.0)))))) - (((data["max_last_10000"]) + (((data["ffti_skew"]) * (data["ffti_num_peaks_50"]))))))) +

            0.033341*np.tanh(((((((data["ffti_c3_100"]) + ((((((data["fftr_spkt_welch_density_10"]) + (data["fftr_c3_5"]))) + ((((data["fftr_c3_5"]) + (((data["fftr_time_rev_asym_stat_5"]) * (data["ffti_exp_Moving_average_30000_std"]))))/2.0)))/2.0)))) * 2.0)) * 2.0)) +

            0.024416*np.tanh(((((data["fftr_hmean"]) / 2.0)) - ((((((data["min_roll_std_50"]) * (((data["min_roll_std_50"]) + (((data["fftr_mean_first_10000"]) + (data["ffti_av_change_abs_roll_std_10000"]))))))) + (data["fftr_percentile_roll_std_10"]))/2.0)))) +

            0.039984*np.tanh(((((data["ffti_percentile_roll_std_25"]) * (((((((data["ffti_num_peaks_100"]) + (data["min_last_50000"]))) + ((((((data["ffti_mean_last_10000"]) + (data["max_first_10000"]))/2.0)) + (data["fftr_percentile_roll_std_10"]))))) * 2.0)))) * 2.0)) +

            0.039984*np.tanh((((((((data["ffti_percentile_roll_mean_80"]) + (data["ffti_min_roll_std_10"]))) * ((-1.0*((data["ffti_av_change_abs_roll_std_10"])))))) + (((data["fftr_spkt_welch_density_10"]) * (((data["ffti_min_roll_std_10"]) - (data["fftr_spkt_welch_density_10"]))))))/2.0)) +

            0.039969*np.tanh(((data["std_last_10000"]) * ((((((data["ffti_mean_change_abs"]) + (data["min_roll_std_500"]))/2.0)) + (((((((data["max_roll_std_500"]) + (data["ffti_percentile_roll_std_10"]))) + (data["av_change_rate_roll_mean_10000"]))) + (data["av_change_rate_roll_mean_10000"]))))))) +

            0.039984*np.tanh((-1.0*((((((data["ffti_percentile_roll_std_60"]) * (((((data["fftr_percentile_80"]) + (((data["fftr_percentile_80"]) + (data["fftr_exp_Moving_average_50000_mean"]))))) + (((data["fftr_num_peaks_100"]) + (data["ffti_exp_Moving_average_30000_mean"]))))))) * 2.0))))) +

            0.039984*np.tanh((((((data["fftr_exp_Moving_average_30000_mean"]) * ((((((data["av_change_rate_roll_std_100"]) + (data["autocorrelation_5"]))/2.0)) - (((data["ffti_ave_roll_mean_1000"]) * (data["ffti_skew"]))))))) + ((((data["autocorrelation_5"]) + (data["ffti_percentile_roll_std_10"]))/2.0)))/2.0)) +

            0.039953*np.tanh((-1.0*((((((((((data["ffti_autocorrelation_10"]) + (data["ffti_percentile_roll_std_70"]))/2.0)) + (data["ffti_percentile_roll_std_75"]))/2.0)) - (((data["fftr_mean_last_1000"]) * ((((data["ffti_num_peaks_100"]) + (data["ffti_autocorrelation_1000"]))/2.0))))))))) +

            0.039969*np.tanh(((data["fftr_percentile_roll_std_5"]) * (((((((data["autocorrelation_5"]) + (data["exp_Moving_average_30000_std"]))) + (((data["fftr_percentile_roll_std_20"]) - (((data["ffti_min_roll_std_10"]) + (data["ffti_min_roll_std_10"]))))))) + (data["ffti_av_change_abs_roll_std_1000"]))))) +

            0.039984*np.tanh(((data["fftr_time_rev_asym_stat_1"]) * (((data["classic_sta_lta8_mean"]) * (((((data["ffti_percentile_30"]) + (((((((data["ffti_sum"]) + (data["max_last_10000"]))/2.0)) + (data["ffti_autocorrelation_10"]))/2.0)))) + (data["ffti_sum"]))))))) +

            0.040000*np.tanh(((((data["ffti_av_change_abs_roll_std_1000"]) * (((((data["ffti_av_change_abs_roll_std_1000"]) - (data["exp_Moving_average_30000_std"]))) + (((((((data["ffti_min_roll_std_10"]) * (data["ffti_av_change_abs_roll_std_1000"]))) * 2.0)) + (data["ffti_autocorrelation_10"]))))))) * 2.0)) +

            0.040000*np.tanh(((data["ffti_max_first_1000"]) * (((((data["ffti_min_roll_std_500"]) - (data["ffti_autocorrelation_50"]))) - (((((((data["ffti_autocorrelation_50"]) - (data["ffti_num_peaks_100"]))) - (data["ffti_num_peaks_10"]))) - (data["av_change_abs_roll_std_500"]))))))) +

            0.037061*np.tanh(((((((((data["ffti_percentile_roll_std_1"]) * (((((data["fftr_exp_Moving_average_300_mean"]) + (((data["ffti_max_last_1000"]) - (data["ffti_av_change_abs_roll_mean_50"]))))) - (data["ffti_av_change_abs_roll_mean_50"]))))) * (data["autocorrelation_10000"]))) * 2.0)) * 2.0)) +

            0.040000*np.tanh(((data["min_roll_std_100"]) * (((((((data["min_roll_mean_500"]) + (data["min_roll_mean_500"]))) + (((data["fftr_min_roll_std_1000"]) + (data["min_first_50000"]))))) - (data["mean_change_rate_last_10000"]))))) +

            0.040000*np.tanh(((data["ffti_mean_change_rate"]) * (((data["fftr_classic_sta_lta1_mean"]) * (((data["classic_sta_lta2_mean"]) + (((data["classic_sta_lta2_mean"]) + ((-1.0*((((data["ave_roll_mean_500"]) - (data["ffti_min_roll_std_1000"])))))))))))))) +

            0.040000*np.tanh(((((data["min_roll_std_50"]) * (((data["autocorrelation_10"]) * (((data["fftr_mean_first_50000"]) - (((((data["ffti_std_roll_mean_10000"]) - (data["fftr_min_roll_std_100"]))) * (data["av_change_rate_roll_mean_10"]))))))))) * 2.0)) +

            0.039922*np.tanh((((((data["ffti_min_roll_std_500"]) * (((((data["fftr_abs_percentile_10"]) * 2.0)) * 2.0)))) + (((data["ffti_mean"]) * ((((-1.0*((data["fftr_num_peaks_50"])))) + ((-1.0*((data["fftr_num_peaks_20"])))))))))/2.0)) +

            0.040000*np.tanh(((data["abs_percentile_75"]) * (((data["min_roll_std_50"]) + (((((data["ffti_min_first_1000"]) * (data["percentile_60"]))) + (((data["ffti_num_peaks_50"]) + (data["fftr_percentile_roll_mean_1"]))))))))) +

            0.039984*np.tanh((((((data["fftr_range_2000_3000"]) + (data["std_roll_mean_10000"]))/2.0)) * (((data["autocorrelation_10000"]) + ((-1.0*((((data["std_roll_mean_10000"]) * (((data["fftr_mean"]) - ((-1.0*((data["percentile_99"]))))))))))))))) +

            0.039578*np.tanh(((((data["ffti_mean_first_1000"]) + (((((data["num_peaks_50"]) - (data["fftr_max_roll_mean_10000"]))) - ((((data["fftr_max_roll_mean_10000"]) + (data["autocorrelation_500"]))/2.0)))))) * (((data["autocorrelation_500"]) + (data["fftr_percentile_roll_std_60"]))))) +

            0.040000*np.tanh(((((((data["ffti_classic_sta_lta1_mean"]) + (data["fftr_percentile_50"]))/2.0)) + (((data["fftr_min_roll_std_1000"]) * (((data["ffti_ave10"]) + (((((data["spkt_welch_density_100"]) + (data["fftr_ave_roll_mean_500"]))) - (data["ffti_sum"]))))))))/2.0)) +

            0.035592*np.tanh(((data["percentile_roll_std_40"]) * (((((((data["autocorrelation_5000"]) + (((data["ffti_av_change_abs_roll_std_50"]) * 2.0)))) + (((data["mean_change_abs"]) + (data["ffti_av_change_abs_roll_std_50"]))))) / 2.0)))) +

            0.039984*np.tanh(((data["ffti_hmean"]) * (((data["min_first_10000"]) + (((data["min_roll_std_50"]) * (((((data["fftr_autocorrelation_10000"]) + (data["mean_first_1000"]))) + (data["fftr_min_roll_std_500"]))))))))) +

            0.040000*np.tanh(((((data["percentile_25"]) * (data["fftr_range_2000_3000"]))) + (((((data["av_change_rate_roll_mean_50"]) * 2.0)) * (((((((data["mean_change_rate_last_1000"]) + (data["skew"]))) + (data["ffti_num_peaks_50"]))) * 2.0)))))) +

            0.040000*np.tanh(((data["min_roll_std_10"]) * (((data["ffti_av_change_abs_roll_mean_10000"]) + (((((data["ffti_av_change_abs_roll_mean_10000"]) * (data["min_roll_std_10"]))) - (((((data["ffti_classic_sta_lta6_mean"]) * (data["exp_Moving_average_50000_std"]))) - (data["exp_Moving_average_50000_std"]))))))))) +

            0.039969*np.tanh(((data["ffti_percentile_roll_std_30"]) * (((((data["ffti_kurt"]) + (data["fftr_time_rev_asym_stat_100"]))) + (((data["exp_Moving_average_3000_mean"]) + (((((data["fftr_time_rev_asym_stat_100"]) - (data["ffti_min_first_50000"]))) - (data["fftr_ave_roll_mean_10000"]))))))))) +

            0.040000*np.tanh(((((((data["av_change_rate_roll_mean_10"]) * (((((((((data["fftr_mean_change_rate_first_1000"]) - ((-1.0*((data["fftr_percentile_roll_std_95"])))))) * 2.0)) * (data["ffti_range_minf_m4000"]))) - (data["ffti_range_p4000_pinf"]))))) * 2.0)) * 2.0)) +

            0.038390*np.tanh(((((data["ffti_av_change_abs_roll_std_50"]) * ((-1.0*((data["ffti_mean_first_1000"])))))) - (((data["ffti_ave10"]) * (((data["ffti_percentile_75"]) + (((data["abs_percentile_20"]) * (data["fftr_range_-2000_-1000"]))))))))) +

            0.039984*np.tanh(((data["fftr_c3_1000"]) - ((((((data["mean_change_abs"]) - (data["av_change_rate_roll_mean_50"]))) + ((((((data["fftr_min_roll_mean_1000"]) + (((data["ffti_skew"]) - (data["fftr_c3_1000"]))))/2.0)) + (data["ffti_av_change_abs_roll_std_500"]))))/2.0)))) +

            0.040000*np.tanh(((data["ffti_abs_max_roll_std_10"]) * (((data["classic_sta_lta4_mean"]) + (((data["ffti_av_change_abs_roll_mean_10000"]) + (((data["classic_sta_lta4_mean"]) + (((data["ffti_autocorrelation_10000"]) + (((data["fftr_percentile_roll_std_95"]) * (data["fftr_autocorrelation_5000"]))))))))))))) +

            0.040000*np.tanh(((data["exp_Moving_average_50000_std"]) * (((data["exp_Moving_average_50000_std"]) * (((data["exp_Moving_average_50000_std"]) * (((((data["fftr_range_-3000_-2000"]) - (data["ffti_percentile_roll_std_60"]))) - (((data["autocorrelation_10000"]) * (data["fftr_range_-3000_-2000"]))))))))))) +

            0.031919*np.tanh((-1.0*(((((-1.0*(((((data["ffti_min_roll_std_100"]) + ((((data["ffti_min_roll_mean_500"]) + (data["ffti_percentile_roll_std_1"]))/2.0)))/2.0))))) + (((data["ffti_ave_roll_mean_1000"]) * (data["ffti_percentile_roll_std_1"])))))))) +

            0.039984*np.tanh((((((data["mean_change_rate_first_1000"]) + ((((((data["fftr_range_-3000_-2000"]) + (((data["mean_change_rate_first_1000"]) * (((data["mean_change_rate_first_1000"]) * (data["med"]))))))/2.0)) * (data["mean_change_rate"]))))/2.0)) * (data["med"]))) +

            0.031153*np.tanh(((((((data["ffti_percentile_roll_mean_90"]) * (((data["av_change_abs_roll_mean_50"]) - (data["av_change_rate_roll_std_10000"]))))) * 2.0)) - (((data["fftr_mean_change_rate_first_10000"]) - (((((data["classic_sta_lta7_mean"]) * (data["ffti_num_peaks_10"]))) * 2.0)))))) +

            0.037515*np.tanh((-1.0*((((data["classic_sta_lta8_mean"]) * (((((data["fftr_min_roll_std_10"]) - (data["ffti_min_roll_mean_10000"]))) - (((data["fftr_std_last_10000"]) * (((((data["ffti_percentile_roll_mean_99"]) * (data["classic_sta_lta8_mean"]))) / 2.0))))))))))) +

            0.039969*np.tanh(((data["ffti_av_change_abs_roll_std_10000"]) * (((data["ffti_ave_roll_mean_10"]) - (((((-1.0*((((((data["ffti_ave_roll_mean_10"]) * (data["ffti_ave_roll_mean_10"]))) - ((-1.0*((data["ffti_min_roll_std_50"]))))))))) + (data["ffti_av_change_abs_roll_std_10000"]))/2.0)))))) +

            0.039406*np.tanh(((((((data["abs_percentile_20"]) * (data["ffti_autocorrelation_1000"]))) - (((data["fftr_min_roll_std_100"]) * (data["classic_sta_lta1_mean"]))))) - (((data["abs_percentile_20"]) * (((data["fftr_min_roll_std_100"]) * (data["classic_sta_lta1_mean"]))))))) +

            0.040000*np.tanh(((data["spkt_welch_density_50"]) * (((((((((data["min_roll_std_50"]) + (data["fftr_mean_change_rate"]))) + ((-1.0*((data["min_roll_std_10000"])))))) + (data["time_rev_asym_stat_10"]))) + (data["fftr_abs_max_roll_mean_500"]))))) +

            0.040000*np.tanh(((data["std_roll_std_10"]) * (((data["fftr_sum"]) * (((((data["ffti_min_roll_std_50"]) - (data["ffti_range_2000_3000"]))) - (((data["fftr_percentile_roll_mean_5"]) * (((data["fftr_percentile_roll_mean_5"]) - (data["ffti_av_change_abs_roll_mean_100"]))))))))))) +

            0.039984*np.tanh(((data["ffti_mean_change_rate"]) * (((((((data["ffti_max_last_1000"]) / 2.0)) - (data["autocorrelation_500"]))) + (((data["av_change_abs_roll_std_500"]) + (((data["fftr_time_rev_asym_stat_10"]) - (data["ffti_min_roll_std_1000"]))))))))) +

            0.040000*np.tanh((((((((((data["ffti_percentile_roll_std_5"]) * 2.0)) + (data["abs_percentile_10"]))/2.0)) * 2.0)) * (((data["ffti_sum"]) - (((data["ffti_av_change_abs_roll_std_10000"]) - (((data["abs_percentile_80"]) - (data["ffti_av_change_abs_roll_std_500"]))))))))) +

            0.040000*np.tanh((((data["ffti_percentile_roll_std_30"]) + (((data["fftr_num_peaks_100"]) * ((((((((data["min_roll_std_1000"]) * (data["ffti_av_change_abs_roll_std_10"]))) * 2.0)) + (data["fftr_num_peaks_100"]))/2.0)))))/2.0)) +

            0.039218*np.tanh(((data["autocorrelation_5000"]) * ((((((data["ffti_autocorrelation_5000"]) + (data["ffti_percentile_roll_std_20"]))/2.0)) + (((data["fftr_std_roll_mean_100"]) * (((((data["ffti_av_change_abs_roll_mean_10"]) / 2.0)) + (((data["ffti_autocorrelation_5000"]) * 2.0)))))))))) +

            0.034201*np.tanh(((data["ffti_ave10"]) * ((-1.0*((((data["autocorrelation_500"]) + ((-1.0*((((((data["fftr_mean_last_50000"]) * (data["fftr_mean_last_50000"]))) + (data["mean_first_10000"]))))))))))))) +

            0.038921*np.tanh(((data["fftr_num_peaks_10"]) * (((data["exp_Moving_average_30000_std"]) - (((((((((data["ffti_mean_change_rate"]) * 2.0)) * 2.0)) * 2.0)) * ((((data["fftr_percentile_roll_mean_1"]) + (data["ffti_mean_change_rate"]))/2.0)))))))) +

            0.039937*np.tanh((((((data["ffti_mean_last_50000"]) + (((data["fftr_percentile_roll_std_40"]) + (((((data["ffti_mean_change_rate_last_10000"]) * (data["ffti_mean_change_rate_last_1000"]))) - ((((data["ffti_min_roll_std_10000"]) + (data["ffti_abs_percentile_5"]))/2.0)))))))) + (data["ffti_Hann_window_mean_15000"]))/2.0)) +

            0.039250*np.tanh(((((((data["ffti_percentile_roll_mean_95"]) * (((data["ffti_exp_Moving_average_300_mean"]) * (((data["ffti_percentile_40"]) + (data["ffti_max_roll_mean_10"]))))))) + (((data["ffti_percentile_40"]) + (data["fftr_abs_percentile_20"]))))) * 2.0)) +

            0.039937*np.tanh((((((-1.0*(((((-1.0*((data["fftr_min_roll_std_500"])))) + (data["percentile_roll_std_1"])))))) * (data["exp_Moving_average_3000_std"]))) + ((((data["classic_sta_lta6_mean"]) + (((data["fftr_min_roll_std_500"]) * (data["ffti_mean"]))))/2.0)))) +

            0.039984*np.tanh(((((((data["num_peaks_50"]) * (data["ffti_mean_first_50000"]))) - (((data["fftr_max_last_10000"]) - (((data["ffti_percentile_roll_mean_80"]) * (data["ffti_min_last_1000"]))))))) + (((data["min_last_1000"]) * (data["ffti_min_last_1000"]))))) +

            0.038312*np.tanh(((((((data["skew"]) * (((((data["ffti_percentile_roll_mean_30"]) + (((data["ffti_percentile_roll_mean_30"]) * (data["ffti_av_change_abs_roll_std_10"]))))) - (data["fftr_range_-3000_-2000"]))))) * 2.0)) * 2.0)) +

            0.039969*np.tanh(((data["av_change_abs_roll_mean_50"]) * ((-1.0*((((((data["num_peaks_50"]) + (data["ffti_av_change_abs_roll_std_500"]))) + (((data["ffti_av_change_abs_roll_std_500"]) + (((data["max_first_1000"]) + (data["ffti_num_peaks_100"])))))))))))) +

            0.040000*np.tanh(((data["med"]) * (((((data["med"]) * (data["fftr_percentile_roll_mean_99"]))) + (((data["ffti_min_first_1000"]) * (((data["fftr_percentile_roll_mean_99"]) + (((data["ffti_med"]) + (data["fftr_percentile_roll_mean_99"]))))))))))) +

            0.039093*np.tanh(((data["classic_sta_lta2_mean"]) * (((((data["fftr_percentile_roll_std_25"]) - (data["spkt_welch_density_50"]))) - (((((((data["fftr_abs_percentile_20"]) + (data["spkt_welch_density_50"]))) * (data["fftr_num_peaks_10"]))) * 2.0)))))) +

            0.039969*np.tanh(((((((data["skew"]) + (((data["ffti_percentile_roll_mean_75"]) * (((data["fftr_moment_3"]) + (data["fftr_autocorrelation_1000"]))))))/2.0)) + (((((data["fftr_percentile_roll_std_20"]) * (data["fftr_abs_percentile_1"]))) * 2.0)))/2.0)) +

            0.039969*np.tanh(((((((((data["abs_percentile_80"]) * ((((data["abs_percentile_10"]) + (data["fftr_classic_sta_lta6_mean"]))/2.0)))) - (data["fftr_percentile_roll_std_70"]))) - (data["fftr_mean_change_rate_last_1000"]))) - (((data["fftr_classic_sta_lta6_mean"]) * (data["fftr_mean_change_rate_last_1000"]))))) +

            0.038812*np.tanh(((data["fftr_autocorrelation_10000"]) * (((((((((data["mean_change_rate_first_1000"]) - (data["autocorrelation_10000"]))) - (data["ffti_percentile_roll_mean_30"]))) - (data["ffti_min_roll_std_10000"]))) - ((((data["min_first_50000"]) + (data["fftr_autocorrelation_10000"]))/2.0)))))) +

            0.039969*np.tanh(((((((data["abs_percentile_10"]) - (data["fftr_mean_change_rate_first_10000"]))) - (((((data["fftr_exp_Moving_average_3000_mean"]) - (data["ffti_mean_change_rate_last_1000"]))) - (((data["exp_Moving_average_50000_std"]) / 2.0)))))) * (data["min_roll_std_10"]))) +

            0.040000*np.tanh(((((data["av_change_abs_roll_std_50"]) * ((((data["fftr_autocorrelation_10000"]) + (((((((((data["ffti_percentile_roll_mean_60"]) - (data["fftr_percentile_roll_std_60"]))) - (data["fftr_percentile_roll_std_60"]))) - (data["fftr_percentile_roll_std_60"]))) * 2.0)))/2.0)))) * 2.0)) +

            0.024369*np.tanh(((data["ffti_spkt_welch_density_5"]) * (((data["spkt_welch_density_50"]) + (((((((((data["ffti_spkt_welch_density_5"]) + (((((data["fftr_percentile_roll_std_5"]) + (data["fftr_percentile_roll_mean_1"]))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)))))) +

            0.040000*np.tanh(((data["ffti_percentile_roll_std_60"]) * (((((data["ffti_min_last_1000"]) + (((data["ffti_max_to_min_diff"]) * 2.0)))) + (((((data["fftr_av_change_abs_roll_mean_500"]) + ((-1.0*((data["ffti_mean"])))))) + (data["fftr_min_roll_std_500"]))))))) +

            0.039969*np.tanh(((data["fftr_percentile_roll_std_40"]) * (((((data["ffti_percentile_roll_std_5"]) + (data["ffti_percentile_roll_std_5"]))) - (((data["std_roll_mean_10000"]) - (((((data["fftr_percentile_roll_std_60"]) + (data["trend"]))) - (data["min_roll_std_10"]))))))))))

X_tr = pd.read_csv('../input/lanl-features/train_features.csv')

X_test = pd.read_csv('../input/lanl-features/test_features.csv')

y_tr = pd.read_csv('../input/lanl-features/y.csv')



alldata = pd.concat([X_tr, X_test])

#alldata.drop('seg_id',axis=1,inplace=True)

alldata["var_larger_than_std_dev"] = alldata["var_larger_than_std_dev"] * 1
scaler = StandardScaler()

alldata = pd.DataFrame(scaler.fit_transform(alldata), columns=alldata.columns)

alldata["gpI"] = GPI(alldata)

alldata["gpII"] = GPII(alldata)

alldata["gpIII"] = GPIII(alldata)
X_tr_scaled = alldata[:X_tr.shape[0]]

X_test_scaled = alldata[X_tr.shape[0]:]
from gplearn.genetic import SymbolicRegressor

from gplearn.functions import make_function
def tanh(x):

    return np.tanh(x);

def sinh(x):

    return np.sinh(x);

def cosh(x):

    return np.cosh(x);



gp_tanh = make_function(tanh,"tanh",1)

gp_sinh = make_function(sinh,"sinh",1)

gp_cosh = make_function(cosh,"cosh",1)
est_gp = SymbolicRegressor(population_size=2000,

                               tournament_size=50,

                               generations=10, stopping_criteria=0.0,

                               p_crossover=0.9, p_subtree_mutation=0.00001, p_hoist_mutation=0.00001, p_point_mutation=0.00001,

                               max_samples=1.0, verbose=1,

                               #function_set = ('add', 'sub', 'mul', 'div', gp_tanh, 'sqrt', 'log', 'abs', 'neg', 'inv','max', 'min', 'tan', 'cos', 'sin'),

                               function_set = (gp_tanh, 'add', 'sub', 'mul', 'div'),

                               metric = 'mean absolute error', warm_start=True,

                               n_jobs = -1, parsimony_coefficient=0.00001, random_state=11)
est_gp.fit(X_tr_scaled, y_tr)
y_gp = est_gp.predict(X_tr_scaled)

gpLearn_MAE = mean_absolute_error(y_tr, y_gp)

print("gpGpiLearn MAE:", gpLearn_MAE)
submission = pd.read_csv('../input/lanl-features/submission_1.csv', index_col='seg_id')

submission.time_to_failure = est_gp.predict(X_test_scaled)

submission.to_csv('submission.csv', index=True)

submission