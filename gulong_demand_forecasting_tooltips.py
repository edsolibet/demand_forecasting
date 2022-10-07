# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:56:33 2022

@author: carlo
"""

tooltips_text = {}

tooltips_text['platform'] = 'Platform to import data.'

tooltips_text['date_columns'] = 'Name of date column in data.'

tooltips_text['group_data'] = 'Parameter how data will be grouped. *Default: model*'

tooltips_text['min_num_trans'] = 'Minimum number of transactions for each group to be included.'

tooltips_text['date_cutoff'] = 'Date cutoff at which the group registers a transaction entry.'


tooltips_text['calib_period_start'] = '''Date of start of calibration period. 
            If None selection, calibration period runs from start of dataset to selected end date.'''

tooltips_text['growth'] = '''**Logistic**: bounded growth/trend and requires input of carrying capacity (cap) at which the data is assumed to be at maximum.
                     **Linear**: Growth/trend characterized by a slope and intercept.'''
                     
tooltips_text['cap_value'] = 'Assumed maximum/ceiling value  at which the data slowly approaches'

tooltips_text['floor_value'] = 'Assumed minimum/floor value at which the data slowly approaches'

tooltips_text['changepoint_select'] = '''**Auto**: Let the model select the appropriate changepoints with maximum of n_changepoints.
                     **Manual**: User inputs dates at which to place changepoints.'''
                     
tooltips_text['n_changepoints'] = '''Changepoints are potential locations at which the trend recognizes a change in the trend or rate. 
                     These changepoints are spaced uniformly within the changepoint_range. *Default: 25-40*'''
                     
tooltips_text['changepoints'] = 'Dates which are expected to see a change in trend. *Format: "YYYY-MM-DD"*'

tooltips_text['changepoint_prior_scale'] = 'Percentage of selected data at which the changepoints can be placed automatically.'

tooltips_text['changepoint_range'] = '''Adjusts trend flexibility/smoothing (overfit or underfit). 
                    Increasing this value increases flexibility (underfit) and vice versa. *Default: 5.0-15.0*'''
                    
tooltips_text['add_seasonality'] = '''**Auto**: sets the default parameter values defined by Prophet. 
                     **True**: Adds seasonality but requires user input for parameter values.
                     **False**: No seasonality'''
                     
tooltips_text['seasonality_mode'] = '''**Additive**: The seasonalities are added to the trend to get the forecast (constant effect). 
                 **Multiplicative**: The seasonalities are multiplied to the trend (values grow with the trend).'''
                 
tooltips_text['set_overall_seasonality_prior_scale'] = 'Set overall prior scale (magnitude of effect) for all seasonalities included.'

tooltips_text['overall_seasonality_prior_scale'] = 'Input overall prior scale for all seasonalities. *Default: 3.0-6.0*'

tooltips_text['add_yearly_seasonality'] = '''**auto**: Selects default parameters for yearly seasonality by Prophet.
                     **False**: No yearly seasonality added to the model.
                     **custom**: User decides input parameters for yearly seasonality. *Period = 365 days*'''
                     
tooltips_text['yearly_order'] = 'Determines how quickly the seasonality can change (higher order = higher frequency changes).'

tooltips_text['yearly_prior_scale'] = 'Smoothing factor for yearly seasonality.'

tooltips_text['add_monthly_seasonality'] = '''**auto**: Selects default parameters for monthly seasonality by Prophet.
                     **False**: No monthly seasonality added to the model.
                     **custom**: User decides input parameters for monthly seasonality. *Period = 30.4 days*'''
                     
tooltips_text['monthly_order'] = 'Determines how quickly the seasonality can change (higher order = higher frequency changes). *Default: 9.0*'

tooltips_text['monthly_prior_scale'] = 'Smoothing factor for monthly seasonality. *Default: 6.0*'

tooltips_text['add_weekly_seasonality'] = '''**auto**: Selects default parameters for weekly seasonality by Prophet.
                     **False**: No weekly seasonality added to the model.
                     **custom**: User decides input parameters for weekly seasonality. *Period = 7 days*'''
                     
tooltips_text['weekly_order'] = 'Determines how quickly the seasonality can change (higher order = higher frequency changes). *Default: 3.0-6.0*'

tooltips_text['weekly_prior_scale'] = 'Smoothing factor for weekly seasonality. *Default: 6.0*'

tooltips_text['add_holidays'] = 'Include holidays in Prophet model.'

tooltips_text['add_public_holidays'] = 'Include public country holidays as defined by Prophet (Christmas, New Years Eve, Holy Week, etc.).'

tooltips_text['add_saved_holidays'] = 'Include saved holidays in model (Fathers Day).'

tooltips_text['holiday_prior_scale'] = 'Smoothing/effect flexibility of holidays.'

tooltips_text['add_metrics'] = '''Include traffic data metrics as factors/regressors used to predict target. 
                                            Any regressors added will require data input during forecast dates.'''

tooltips_text['add_metrics_select'] = '*Default*: ctr_total, ad_costs_total'

tooltips_text['add_google_trends'] = 'Include google trend data as regressors. Any regressors added will require data input during forecast dates.'

tooltips_text = {'platform_select': 'Platform data to use',
                 'date_column': 'Name of date column in data',
                 'target_column': 'Metric or value to forecast',
                 'training_start': 'Start date of training data to train model',
                 'training_end': 'End date of training data to train model',
                 'forecast_checkbox': 'Select whether to predict values for future dates or only include training data',
                 'forecast_horizon': 'Number of days from end of training data to predict',
                 'group_data': 'Parameter how data will be grouped. *Default: model*',
                 'group_select': 'Select which group to forecast.',
                 'min_num_trans': 'Minimum number of transactions for each group to be included.',
                 'date_cutoff': 'Date cutoff at which the group registers a transaction entry.',
                 'calib_period_start': '''Date of start of calibration period. 
            If None selection, calibration period runs from start of dataset to selected end date.''',
                 'growth': '''**Logistic**: bounded growth/trend and requires input of carrying capacity (cap) at which the data is assumed to be at maximum.
                     **Linear**: Growth/trend characterized by a slope and intercept.''',
                 'cap_value': 'Assumed maximum/ceiling value  at which the data slowly approaches',
                 'floor_value': 'Assumed minimum/floor value at which the data slowly approaches',
                 'changepoint_select': '''**Auto**: Let the model select the appropriate changepoints with maximum of n_changepoints.
                     **Manual**: User inputs dates at which to place changepoints.''',
                 'n_changepoints': '''Changepoints are potential locations at which the trend recognizes a change in the trend or rate. 
                     These changepoints are spaced uniformly within the changepoint_range. *Default: 25-40*''',
                 'changepoints': 'Dates which are expected to see a change in trend. *Format: "YYYY-MM-DD"*',
                 'changepoint_range': 'Percentage of selected data at which the changepoints can be placed automatically.',
                 'changepoint_prior_scale': 'Adjusts trend flexibility/smoothing (overfit or underfit). Increasing this value increases flexibility (underfit) and vice versa. *Default: 5.0-15.0*',
                 'add_seasonality': '''**Auto**: sets the default parameter values defined by Prophet. 
                     **True**: Adds seasonality but requires user input for parameter values.
                     **False**: No seasonality''',
                 'seasonality_mode': '''**Additive**: The seasonalities are added to the trend to get the forecast (constant effect). 
                 **Multiplicative**: The seasonalities are multiplied to the trend (values grow with the trend).''',
                 'set_overall_seasonality_prior_scale': 'Set overall prior scale (magnitude of effect) for all seasonalities included.',
                 'overall_seasonality_prior_scale': 'Input overall prior scale for all seasonalities. *Default: 3.0-6.0*',
                 'add_yearly_seasonality': '''**auto**: Selects default parameters for yearly seasonality by Prophet.
                     **False**: No yearly seasonality added to the model.
                     **custom**: User decides input parameters for yearly seasonality. *Period = 365 days*''',
                 'yearly_order': 'Determines how quickly the seasonality can change (higher order = higher frequency changes).',
                 'yearly_prior_scale': 'Smoothing factor for yearly seasonality.',
                 'add_monthly_seasonality': '''**auto**: Selects default parameters for monthly seasonality by Prophet.
                     **False**: No monthly seasonality added to the model.
                     **custom**: User decides input parameters for monthly seasonality. *Period = 30.4 days*''',
                 'monthly_order': 'Determines how quickly the seasonality can change (higher order = higher frequency changes). *Default: 9.0*',
                 'monthly_prior_scale': 'Smoothing factor for monthly seasonality. *Default: 6.0*',
                 'add_weekly_seasonality': '''**auto**: Selects default parameters for weekly seasonality by Prophet.
                     **False**: No weekly seasonality added to the model.
                     **custom**: User decides input parameters for weekly seasonality. *Period = 7 days*''',
                 'weekly_order': 'Determines how quickly the seasonality can change (higher order = higher frequency changes). *Default: 3.0-6.0*',
                 'weekly_prior_scale': 'Smoothing factor for weekly seasonality. *Default: 6.0*',
                 'add_holidays': 'Include holidays in Prophet model.',
                 'add_public_holidays': 'Include public country holidays as defined by Prophet (Christmas, New Years Eve, Holy Week, etc.).',
                 'add_saved_holidays': 'Include saved holidays in model (Fathers Day).',
                 'holiday_prior_scale': 'Smoothing/effect flexibility of holidays.',
                 'add_metrics': 'Include traffic data metrics as factors/regressors used to predict target. Any regressors added will require data input during forecast dates.',
                 'add_metrics_select': '*Default*: ctr_total, ad_costs_total',
                 'add_google_trends': 'Include google trend data as regressors. Any regressors added will require data input during forecast dates.',
                 'gtrend_kw': 'Input google trends keywords separated by a space.',
                 'data_input_type': 'Type of regressor data to input.',
                 'data_input_total': 'Input the total value of the regressor over the selected forecast dates. Total is divided equally for each day.',
                 'data_input_average': 'Input the average value of the regressor over the selected forecast dates. Each day will have the same value.',
                 'add_custom_regressors': 'Include saved regressors (based on dates).',
                 'nan_clean_method': '''**fill with zero**: Replace NaNs with 0 (zero).
                     **fill with adjacent mean**: Fill in NaNs with average of before and after values.
                     **remove rows with NaNs**: Removes rows with NaNs from the dataset.''',
                 'outliers': 'Outliers are data points which are outside the normal range with respect to the neighboring data or as a whole.',
                 'transform': '**Moving average**: Average of data over a certain range of observations or window.',
                 'window': 'How many observations we have to take for the calculation of the moving average.',
                 'remove_outlier_method': '''**KNN**: K-Nearest Neighbors; Finds outliers based on nearby datapoints.
                     **LOF**: Local Outlier Factor; Calculates the local density deviation of a given data point wrt to its neighbors.
                     **Isolation Forest**: Uses a decision tree algorithm to determine data anomaly which is outside the normality in terms of tree path length.''',
                 'KNN_neighbors': 'K-Nearest neighbors number of neighbors to consider.',
                 'IF_estimators': 'Number of base estimators used in IsolationForest Algorithm.',
                 'IF_max_samples': 'Number of samples to draw from data to train each estimator.',
                 'mae': 'Mean Absolute Error; measure of errors between paired observations. Acceptable value is 10% of the average of the data.',
                 'rmse' : 'Root Mean Square Error; one of the most commonly used measures for evaluating the quality of predictions using Euclidean distance. Acceptable value is 10% of the average of the data.',
                 'mape': 'Mean Absolute Percentage Error; measures accuracy in terms of percentage. MAPE < 25% indicates an acceptable forecast.',
                 'pearson_coeff': '''Pearson correlation coefficient (R<sup>2</sup>)  measures the strength of relationship between two variables.
                    A value of > 0.9 suggestions a **strong**, **positive** association between two variables.
                    ''',
                 'overview': '''This plot shows the model fitting on the training data and forecasted values (if any). Black points are the actual/transformed values.
                 Thick blue line is the forecast value, light blue bands are the upper and lower bounds of the forecasted value. Vertical orange lines are the locations of the changepoints.''',
                 'forecast_vs_actual': '''This plot shows the forecasted values in relation to the actual values. Each point corresponds to a single date from the dataset. 
                 The red line indicates the optimal result for each point. ''',
                 'impact_of_components': 'Each plot shows how the corresponding seasonality, regressor, or holiday affects the predicted result with respect to if the component was not present in the model.'}

