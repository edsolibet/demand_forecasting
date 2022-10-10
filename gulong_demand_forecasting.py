# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 20:31:25 2022

@author: carlo
"""

# Library Install & Import
# =============================================================================

# Essential/Data import
import numpy as np
import pandas as pd
import datetime as dt
from datetime import timedelta
import openpyxl, requests, string, re
from io import BytesIO

# streamlit
import streamlit as st

# Modelling and Forecasting
# =============================================================================
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.utilities import regressor_coefficients
# Customer transaction behavior
from lifetimes.fitters.pareto_nbd_fitter import ParetoNBDFitter
# from lifetimes import GammaGammaFitter
from pytrends.request import TrendReq
import plotly.express as px
import plotly.graph_objects as go

# Machine Learning libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score


import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
from joblib import dump, load

# custom modules
from gulong_demand_forecasting_tooltips import tooltips_text

# ===============================================

def ratio(a, b):
    '''
    Function for calculating ratios to avoid inf
    '''
    return a/b if b else 0 

def remove_emoji(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, ' ', text).strip()

def fix_name(name):
  '''
  Fix names which are duplicated.
  Ex. "John Smith John Smith"
  
  Parameters:
  -----------
  name: str
    
  Returns:
  --------
  fixed name; str
  
  '''
  name_list = list()
  # removes emojis and ascii characters (i.e. chinese chars)
  name = remove_emoji(name).encode('ascii', 'ignore').decode()
  # split name by spaces
  for n in name.split(' '):
    if n not in name_list:
    # check each character for punctuations minus '.' and ','
      name_list.append(''.join([ch for ch in n 
                                if ch not in string.punctuation.replace('.', '')]))
    else:
        continue
  return ' '.join(name_list).strip()

def cleanup_specs(specs, col):
    '''
    Parameters
    ----------
    specs : string
        String to obtain specs info
    col : string
        Column name to apply function ('width', 'aspect_ratio', 'diameter')

    Returns
    -------
    specs : string
        Corresponding string value in specs to "col"

    '''
    specs_len = len(specs.split('/'))
    if specs_len == 1:
        return specs.split('/')[0]
    else:
        if col == 'width':
            return specs.split('/')[0]
        elif col == 'aspect_ratio':
            if specs.split('/')[1] == '0' or specs.split('/')[1] == '':
                return 'R'
            else:
                return specs.split('/')[1]
        elif col == 'diameter':
            return specs.split('/')[2][1:3]
        else:
            return specs

def combine_specs(row):
    '''
    Helper function to join corrected specs info

    Parameters
    ----------
    row : dataframe row
        From gogulong dataframe
    Returns
    -------
    string
        joined corrected specs info

    '''       
    if '.' in str(row['aspect_ratio']):
        return '/'.join([str(row['width']), str(float(row['aspect_ratio'])), str(row['diameter'])])
    else:
        return '/'.join([str(row['width']), str(row['aspect_ratio']), str(row['diameter'])])

@st.experimental_memo
def import_cleanup_df(platform = "Gulong.ph"):
    # Data Import
    # =============================================================================
    if platform == "Gulong.ph":
        # Website/Traffic ads
        sheet_id = "17Yb2nYaf_0KHQ1dZBGBJkXDcpdht5aGE"
        sheet_name = 'summary'
        url = "https://docs.google.com/spreadsheets/export?exportFormat=xlsx&id=" + sheet_id
        res = requests.get(url)
        data = BytesIO(res.content)
        xlsx = openpyxl.load_workbook(filename=data)
        df_traffic = pd.read_excel(data, sheet_name = sheet_name)
        
        # Transactions
        gulong_txns_redash = "http://app.redash.licagroup.ph/api/queries/104/results.csv?api_key=YqUI9o2bQn7lQUjlRd9gihjgAhs8ls1EBdYNixaO"
        df_txns_ = pd.read_csv(gulong_txns_redash , parse_dates = ['date'])
        
        # daily traffic
        df_traffic = df_traffic.dropna(axis=1)
        df_traffic = df_traffic.rename(columns={'Unnamed: 0': 'date'})
        clicks_cols = ['link_clicks_ga', 'link_clicks_fb']
        impressions_cols = ['impressions_ga', 'impressions_fb']
        df_traffic.loc[:, 'clicks_total'] = df_traffic.loc[:,clicks_cols].sum(axis=1)
        df_traffic.loc[:, 'impressions_total'] = df_traffic.loc[:,impressions_cols].sum(axis=1)
        df_traffic.loc[:, 'signups_total'] = df_traffic.loc[:,'signups_ga'] + df_traffic.loc[:, 'signups_backend']
        df_traffic.loc[:, 'ctr_ga'] = df_traffic.apply(lambda x: ratio(x['link_clicks_ga'], x['impressions_ga']), axis=1)
        df_traffic.loc[:, 'ctr_fb'] = df_traffic.apply(lambda x: ratio(x['link_clicks_fb'], x['impressions_fb']), axis=1)
        df_traffic.loc[:, 'ctr_total'] = df_traffic.apply(lambda x: ratio(x['clicks_total'], x['impressions_total']), axis=1)
        df_traffic.loc[:, 'ad_costs_total'] = df_traffic.loc[:, 'ad_costs_ga'] + df_traffic.loc[:, 'ad_costs_fb_total']
        
        purchases_backend_cols = [col for col in df_traffic.columns if 'purchases_backend' in col]
        df_traffic.loc[:, 'purchases_backend_total'] = df_traffic.loc[:,purchases_backend_cols].sum(axis=1)
        df_traffic.loc[:, 'purchases_backend_marketplace'] = df_traffic.loc[:, 'purchases_backend_fb'] + df_traffic.loc[:, 'purchases_backend_shopee'] + df_traffic.loc[:, 'purchases_backend_lazada']
        df_traffic.loc[:, 'purchases_backend_b2b'] = df_traffic.loc[:, 'purchases_backend_b2b'] + df_traffic.loc[:, 'purchases_backend_walk-in']
        df_traffic.drop(labels = ['purchases_backend_shopee', 'purchases_backend_lazada', 
                                     'purchases_backend_fb', 'purchases_backend_walk-in', 
                                     'purchases_backend_nan'], axis=1, inplace=True)
        
        
        #------------------------------------------------------------------------------
        
        # remove cancelled, rejected, unconfirmed order statuses
        #df_txns_ = df_txns_.loc[(df_txns_.status.notnull()) & (df_txns_.cancel_type != 'gulong cancellation') 
        #                & (df_txns_.status.isin(['rejected', 'unconfirmed', 'cancelled']) == False)].reset_index().copy()
        df_txns_ = df_txns_[df_txns_.status == 'fulfilled'].drop_duplicates(subset= ['date', 'make', 'customer_type', 'cost'], keep='first')
        df_txns_.loc[:, 'name'] = df_txns_.apply(lambda x: fix_name(x['name']), axis=1)
        df_txns_.loc[:, 'year-month'] = df_txns_['date'].map(lambda x: x.strftime('%Y-%b'))
        df_txns_.loc[:, 'date'] = pd.to_datetime(df_txns_.loc[:,'date'])
        df_txns_.loc[:, 'month'] = df_txns_.loc[:,'date'].dt.month
        df_txns_.loc[:, 'month_day'] = df_txns_.loc[:,'date'].dt.day
        df_txns_.loc[:, 'week_number'] = df_txns_.loc[:,'date'].dt.strftime('%U')
        df_txns_.loc[:, 'wom'] = df_txns_.loc[:, 'date'].apply(lambda d: (d.day-1) // 7 + 1)
        df_txns_.loc[:, 'weekday'] = df_txns_.loc[:,'date'].dt.dayofweek + 1
        # cleanup dimensions
        df_txns_.loc[:, 'width'] = df_txns_.apply(lambda x: cleanup_specs(x['dimensions'], 'width'), axis=1)
        df_txns_.loc[:, 'aspect_ratio'] = df_txns_.apply(lambda x: cleanup_specs(x['dimensions'], 'aspect_ratio'), axis=1)
        df_txns_.loc[:, 'diameter'] = df_txns_.apply(lambda x: cleanup_specs(x['dimensions'], 'diameter'), axis=1)
        df_txns_.loc[:, 'dimensions'] = df_txns_.apply(lambda x: combine_specs(x), axis=1)
        
        
        columns = ['date', 'customer_type', 'status', 'model', 'make', 'dimensions', 
                   'quantity', 'price', 'discount_amount', 'cost', 'tire_type', 'is_active']
        df_txns = df_txns_[columns].copy()
        return df_traffic.reset_index(), df_txns
    else:
        pass

def plot_graphs(df_temp): 
    fig = go.Figure()
    for model in df_temp['model'].unique():
      df_plot = df_temp.loc[df_temp['model'] == model]
      fig.add_trace(
          go.Scatter(
              x = df_plot['date'],
              y = df_plot['price'],
              mode='lines+markers',
              name = model
          )
      )
    fig.update_yaxes(title_text = 'Price (PHP)',showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_xaxes(title_text = 'Date',showline=True, linewidth=1, linecolor='black', mirror=True,showticklabels=False)
    fig.update_layout(template = 'simple_white', title={'text': "Historical Prices Gulong.ph",
            'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})
    fig.show()


def get_rfm_2(df, start_date = None, end_date = None):
    '''
    Calculates recency, frequency, T, last transaction, and inter-transaction 
    dates for given SKU time series
    
    Parameters
    ----------
    df: dataframe
        transaction dataframe with 'date' as column
    start_date: date, optional, default = None
        starting date of created dataframe
    end_date: date, optional, default = None
        end date of created dataframe
    
    Returns
    -------
    temp_df : dataframe
        transaction dataframe with behavior calcs
    
    '''
    # create placeholder dataframe
    temp_df = pd.DataFrame()
    # establish start and end dates of temp_df
    start = start_date if start_date is not None else df.date.min()
    end = end_date if end_date is not None else df.date.max()
    temp_df['date'] = pd.date_range(start = start,
                                    end = end, freq='D').array
    # frequency
    # cases:
    # 0 if dates before first transaction
    # same frequency for each succeeding date after transaction
    temp_df['freq'] = temp_df.apply(lambda x: 1 if x['date'] in df.date.dt.date.array else 0, 1)
    temp_df['freq'] = temp_df['freq'].cumsum()
    
    # T
    # if date does not start with a transaction, T = 0 until date of trans
    # If f > 0, T increments per day
    last_t = 0
    last_t_list = []
    for f in temp_df['freq']:
        if f > 0:
            last_t_list.append(last_t)
            last_t += 1
        else:
            last_t = 0
            last_t_list.append(last_t)
    #temp_df['T'] = np.array(range(0, len(temp_df)))
    temp_df['T'] = np.array(last_t_list)
    
    # last_txn
    # 0 until first transaction, increments by 1 for each day w/o new transaction
    # resets to 0 every new transaction
    last_txn, last_f = 0, -1
    last_txn_list = []
    for f in temp_df['freq']:
        if f == last_f and f > 0:
        # if start_date does not start with a transaction
            last_txn += 1
        else:
            last_txn = 0
        last_txn_list.append(last_txn)
        last_f = f
    temp_df['last_txn'] = np.array(last_txn_list)
    
    # recency
    last_f, last_recency, last_index = 0, -1, 0
    recency = []
    for index, f in enumerate(temp_df['freq']):
        if f != last_f and f == 1:
            # first transaction
            # recency = 0, record index of first trans
            last_index = index 
            last_recency = 0
        elif f != last_f and f > 1:
            # new transaction (not first)
            last_recency = index - last_index
        elif f == last_f and f > 0:
            # rows after new transaction
            pass
        else:
            # rows before first transaction
            last_recency = 0
        recency.append(last_recency)
        last_f = f
    temp_df['recency'] = np.array(recency)
    
    # ITT
    temp_df['ITT'] = temp_df.apply(lambda x: round(x['recency']/x['freq'], 2) if x['freq'] else 0, 1)
    
    # total_qty
    temp_df['total_qty'] = temp_df.apply(lambda x: df[df.date==x['date']]['total_qty'].values.sum() if x['date'] in df.date.dt.date.array else 0, 1)
    temp_df['total_sales'] = temp_df.apply(lambda x: df[df.date==x['date']]['total_sales'].values.sum() if x['date'] in df.date.dt.date.array else 0, 1)
    return temp_df
    
@st.experimental_singleton
def fit_models(df, penalizer = 0.001):
    '''
    Fit Pareto/NBD model to transaction data
    
    Parameters
    ----------
    df: dataframe
        transaction data frame, grouped by date
        
    Returns
    -------
    pnbd: model
        Fitted pareto/nbd model
    '''
    pnbd = ParetoNBDFitter(penalizer_coef=penalizer)
    pnbd.fit(df['freq'], df['recency'], df['T'])
    return pnbd

def lifetimes_stats(_pnbd, t, df):
    '''
    Parameters
    ----------
    _pnbd : fitted Pareto model
    t: int
        time in days to predict
    df : dataframe
        dataframe with unique date column
    
    Returns
    -------
    df : dataframe
        
    
    '''
    # calculate probability of active
    df.loc[:,'prob_active'] = df.apply(lambda x: _pnbd.conditional_probability_alive(x['freq'], x['recency'], x['T'])*100, 1)
    df.loc[:, 'expected_purchases'] = df.apply(lambda x: 
            _pnbd.conditional_expected_number_of_purchases_up_to_time(t, x['freq'], x['recency'], x['T']),1)
    return df


def get_agg_data(df):
    '''
    Parameters
    ----------
    df : dataframe
        Prepared customer transaction dataframe, ungrouped

    Returns
    -------
    agg_data : dict
        behavior data

    '''
    agg_data = df.groupby('date').agg(freq = ('date', lambda x: 1),
                                  total_qty = ('quantity', sum),
                                  total_sales = ('cost', sum))
    agg_data.loc[:, 'freq'] = agg_data.freq.cumsum()
    return agg_data.reset_index()

def make_forecast_dataframe(start, end):
    '''
    Creates training dataframe and future dataframe
    
    Parameters
    ----------
    train: dataframe
        Training data set
    end: string
        Ending date of forecast interval
    
    Returns
    -------
    
    '''
    dates = pd.date_range(start=start, end=end, freq='D')
    df = pd.DataFrame(dates).rename(columns={0:'ds'})
    return df

def plot_regressors(evals, selected_regressors):
    '''
    Create plotly chart for selected regressors to preview
    
    Parameters
    ----------
    evals: dataframe
        cleaned dataframe containing date and regressors
    selected_regressors: list
        list of selected regressors (strings) to preview
        
    Returns:
    --------
    None. Creates plotly plots on streamlit for large-magnitude and small-magnitude regressors
    
    '''
    def go_scat(name, x, y):
        go_fig = go.Scatter(name = name,
                            x = x,
                            y = y,
                            mode = 'lines+markers',
                            marker = dict(size=6))
        return go_fig
    
    small_mag = [sel for sel in selected_regressors if np.max(evals[sel]) <= 10]
    large_mag = [sel for sel in selected_regressors if sel not in small_mag]
    
    if len(large_mag) > 0:
        with st.expander('Large-magnitude regressors'):
            go_fig_l = [go_scat(lm, x=evals.ds, y=evals[lm]) for lm in large_mag]
            
            fig_l = go.Figure(go_fig_l)
            
            if len(small_mag) == 1:
                ylabel = large_mag[0]
            else:
                ylabel = "y"
            
            fig_l.update_layout(
                yaxis_title= ylabel,
                hovermode="x",
                height = 600,
                width = 1200,
                legend=dict(orientation='h',
                            yanchor='bottom',
                            y=-0.15,
                            xanchor='left',
                            x=0))
            # Change grid color and axis colors
            fig_l.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='#696969')
            fig_l.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='#696969')
            st.plotly_chart(fig_l, use_container_width = True)
        
    if len(small_mag) > 0:
        with st.expander('Small-magnitude regressors'):
            go_fig_s = [go_scat(sm, x=evals.ds, y=evals[sm]) for sm in small_mag]
            
            fig_s = go.Figure(go_fig_s)
            
            if len(small_mag) == 1:
                ylabel = small_mag[0]
            else:
                ylabel = "y"
            
            fig_s.update_layout(
                yaxis_title = ylabel,
                hovermode="x",
                height = 600,
                width = 1200,
                legend=dict(orientation='h',
                            yanchor='bottom',
                            y=-0.15,
                            xanchor='left',
                            x=0))
            # Change grid color and axis colors
            fig_s.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='#696969')
            fig_s.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='#696969')
            st.plotly_chart(fig_s, use_container_width=True)

def plot_forecast_vs_actual_scatter(evals, forecast):
    '''
    Create a plot for forecasted values vs actual data
    
    Parameters
    ----------
    evals : dataframe
        training dataframe
    forecast: dataframe
        output dataframe from model.predict
    
    Returns
    -------
    fig : figure
        plotly figure
    '''
    evals_ = evals[['ds', 'y']]
    evals_df = pd.concat([evals_, forecast['yhat']], axis=1)
    fig = px.scatter(evals_df,
                     x = 'y',
                     y = 'yhat',
                     opacity=0.5,
                     hover_data={'ds': True, 'y': ':.4f', 'yhat': ':.4f'})
    
    fig.add_trace(
        go.Scatter(
            x=evals_df['y'],
            y=evals_df['y'],
            name='optimal',
            mode='lines',
            line=dict(color='red', width=1.5)))
    
    fig.update_layout(
        xaxis_title="Truth", yaxis_title="Forecast", legend_title_text="", height=450, width=800)
    
    return fig

def convert_csv(df):
    # IMPORTANT: Cache the conversion to prevent recomputation on every rerun.
    return df.to_csv().encode('utf-8')

def remove_neg_val(yhat, yhat_lower, yhat_upper):
    yhat = yhat if yhat > 0 else 0
    yhat_lower = yhat_lower if yhat_lower > 0 else 0
    yhat_upper = yhat_upper if yhat_upper > 0 else 0
    return yhat, yhat_lower, yhat_upper


if __name__ == "__main__":
    # 1. import datasets
    st.sidebar.markdown('# 2. Data Preparation')
    with st.sidebar.expander("Data Selection"):
        # choose platform for data import
        # platform = st.selectbox("Platform",
        #                         options=["Gulong.ph", "Mechanigo.ph"],
        #                         index=0,
        #                         help = tooltips_text['platform_select'])
        df_traffic, df_txns = import_cleanup_df("Gulong.ph")
        
        # date column
        date_col = st.selectbox('Date column', df_traffic.columns[df_traffic.dtypes=='datetime64[ns]'],
                                index=0,
                                help = tooltips_text['date_column'])
        
        # platform_data = {'Gulong.ph': ('quantity', 'sessions', 'purchases_backend_website'),
        #          'Mechanigo.ph': ('sessions', 'bookings_ga')}
        # target column
        # param = st.selectbox('Target column:', platform_data[platform],
        #                         index=0,
        #                         help = tooltips_text['target_column'])
        param = "total_qty"
        
        filter_by = st.selectbox("Group data by",
                                 options = ["model", "make", "dimensions"],
                                 index = 0,
                                 help = tooltips_text['group_data'])
        
        # 2. filter data sets
    with st.sidebar.expander('Filtering'):
        
        # filter by following parameters
        # minimum number of transactions in dataset according to grouping
        min_num_txns = st.number_input(label="Min. number of transactions",
                        min_value = 5,
                        value = 10,
                        help = tooltips_text['min_num_trans'])
        
        # Transaction date cutoff
        max_date = st.date_input(label = "Last transaction date cutoff",
                                 min_value = df_txns['date'].min().date(),
                                 value = pd.to_datetime('2022-03-01'),
                                 help = tooltips_text['date_cutoff'])
        
        # filter models by number of transactions & availability
        data_agg = df_txns.groupby(filter_by) \
                                .agg(min_date=("date", lambda x: x.min().date()),
                                     max_date=("date", lambda x: x.max().date()),
                                     num_txns = ("date", lambda x: len(x))) \
                                .sort_values('num_txns', ascending=False)
        
        # candidate groups after filtering
        select_groups = list(data_agg[(data_agg.num_txns >= min_num_txns) \
                                      & (data_agg.max_date >= max_date)] \
                                .sort_values('num_txns', ascending=False).index)
        
        st.info(f'''Found **{len(select_groups)}** groups.''')
        
        group_selected = st.selectbox(label='Select item to forecast',
                         options=select_groups,
                         index = 0,
                         help = tooltips_text['group_select'])
        
        
        # fit pnbd model on all dataset
        # should only run once (st.experimental_memo)
        pnbd_model = fit_models(get_rfm_2(get_agg_data(df_txns)))
        
        # 3. Create RFM dataframe
        # Use all filtered data for fitting pareto model
        sku_dict = {}
        # groupby date + calc agg data + pareto results
        temp = df_txns[df_txns[filter_by] == group_selected] \
                                .sort_values('date', ascending=True).reset_index()
        
        # generate time series df of only transaction dates
        sku_list = []
        # date not yet restricted to properly get freq, recency, last_txn, ITT
        temp_data = get_rfm_2(get_agg_data(temp))
        
        # 4. Apply Pareto NBD model on calib and obs data
        # sku_dict[item]['pareto_model'] = fit_models(sku_dict[item]['calib'])
        # t = 1 to only predict up to 1 day
        sku_dict[group_selected] = lifetimes_stats(_pnbd = pnbd_model, 
                                             t = 1, 
                                            df = temp_data)
        
    with st.sidebar.expander("Training and Forecast Dataset"):
        st.markdown('### Training dataset')
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            calib_period_start = st.date_input('Training data start date',
                                        value = pd.to_datetime('2022-03-01'),
                                        min_value=sku_dict[group_selected].date.min().date(),
                                        max_value=sku_dict[group_selected].date.max().date()-timedelta(days=30),
                                        help = tooltips_text['training_start'])
        with tcol2:
            calib_period_end = st.date_input('Training data end date',
                                        value = df_traffic.date.max().date(),
                                        min_value= calib_period_start + timedelta(days=30),
                                        max_value= df_traffic.date.max().date(),
                                        help = tooltips_text['training_end'])
        if calib_period_start >= calib_period_end:
            st.error('Train_end should come after train_start.')
        
        if pd.Timestamp(calib_period_start) <= df_traffic.date.max():
            date_series = make_forecast_dataframe(calib_period_start, calib_period_end)
            param_series =  sku_dict[group_selected][sku_dict[group_selected] \
                                                     .date.isin(date_series.ds.values)]['total_qty'] \
                                                     .reset_index()
            evals = pd.concat([date_series, param_series], axis=1) \
                                                     .rename(columns={0: 'ds', 
                                                                      param:'y'}) \
                                                     .drop('index', axis=1)
            # check for NaNs
            if evals.y.isnull().sum() > 0.5*len(evals):
                st.warning('Evals data contains too many NaN values')
                st.dataframe(evals)
        else:
            st.error('val_end is outside available dataset.')
        
        # show calibration period
        st.markdown(f'''Training period: \n 
                    {calib_period_start} to {calib_period_end}''')
        
        st.markdown('### Forecast dataset')
        make_forecast_future = st.checkbox('Make forecast on future dates',
                                           value = True,
                                           help = tooltips_text['forecast_checkbox'])
        if make_forecast_future:
            forecast_end = st.date_input('Forecast horizon (days)',
                                               min_value = calib_period_end + timedelta(days=1),
                                               value = calib_period_end + timedelta(days=15),
                                               help = tooltips_text['forecast_horizon'])
            
            forecast_horizon = (forecast_end - calib_period_end).days + 1
            future = make_forecast_dataframe(start=calib_period_start, 
                                             end=forecast_end)
            st.info(f'''Forecast dates:\n 
                    {calib_period_end+timedelta(days=1)} to {forecast_end}''')
        

    # 5. Use FB Prophet model to RFM dataset
    # MODELLING
    # ========================================================================
    st.sidebar.markdown('# 2. Modelling')
    # default parameters for target cols
    default_params = {
              'total_qty':{'growth': 'logistic',
              'seasonality_mode': 'multiplicative',
              'changepoint_prior_scale': 8.0,
              'n_changepoints' : 30,
              'cap' : 50.0,
              }
              }
    
    params = {}
    with st.sidebar.expander('Model and Growth type'):
        
        growth_type = st.selectbox('growth',
                                   options=['logistic', 'linear'],
                                   index = 0,
                                   help = tooltips_text['growth'])
        
        params['growth'] = growth_type
        if growth_type == 'logistic':
            # if logistic growth, cap value is required
            # cap value is round up to nearest 500
            cap = st.number_input('Fixed cap value',
                                  min_value = 0.0,
                                  max_value = None,
                                  value = np.ceil(max(sku_dict[group_selected][param])/500)*500,
                                  step = 0.01,
                                  help = tooltips_text['cap_value'])
            evals.loc[:, 'cap'] = cap
            # if forecast future, also apply cap to future df
            if make_forecast_future:
                future.loc[:,'cap'] = cap
            
            # floor is optional
            use_floor = st.checkbox('Add floor value')
            if use_floor:
                floor = st.number_input('Fixed floor value',
                                      min_value = 0.0,
                                      max_value = None,
                                      value = 0.0,
                                      step = 0.01,
                                      help = tooltips_text['floor_value'])
                evals.loc[:, 'floor'] = floor
                if make_forecast_future:
                    future.loc[:,'floor'] = floor
            
                # check viability of cap value
                if cap <= floor:
                    st.error('Cap value should be greater than floor value')
    
    
    # CHANGEPOINTS
    # =========================================================================
    with st.sidebar.expander('Changepoints'):
        
        changepoint_select = st.selectbox('Changepoint selection',
                                          options=['Auto', 'Manual'],
                                          index=0,
                                          help = tooltips_text['changepoint_select'])
        
        if changepoint_select == 'Auto':
            n_changepoints = st.slider('Number of changepoints',
                                       min_value = 5,
                                       max_value = 100,
                                       value = default_params[param]['n_changepoints'],
                                       step = 5,
                                       help = tooltips_text['n_changepoints'])
            
            params['n_changepoints'] = n_changepoints
            params.pop('changepoints', None)
            
        elif changepoint_select == 'Manual':
            changepoints = st.multiselect('Select dates to place changepoints',
                                          options = evals.ds.dt.date.tolist(),
                                          default = [evals.ds.dt.date.min(), evals.ds.dt.date.max()],
                                          help = tooltips_text['changepoints'])
            params['changepoints'] = changepoints
            params.pop('n_changepoints', None)
            
        changepoint_prior_scale = st.number_input('changepoint_prior_scale',
                                    min_value=0.05,
                                    max_value=50.0,
                                    value= float(default_params[param]['changepoint_prior_scale']),
                                    step=0.05,
                                    help = tooltips_text['changepoint_prior_scale'])
        changepoint_range = st.number_input('changepoint_range',
                                    min_value=0.05,
                                    max_value=1.0,
                                    value=0.9,
                                    step=0.05,
                                    help = tooltips_text['changepoint_range'])
        
        # add selected inputs to params dict
        
        params['changepoint_prior_scale'] = changepoint_prior_scale
        params['changepoint_range'] = changepoint_range
        
    # apply params to model
    model = Prophet(**params)  # Input param grid
    
    # SEASONALITIES
    # ========================================================================
    with st.sidebar.expander('Seasonalities'):
        season_model = st.selectbox('Add seasonality', 
                            options = ['Auto', 'True', 'False'],
                            index = 1,
                            key = 'season_model',
                            help = tooltips_text['add_seasonality'])
        
        seasonality_scale_dict = {}
        
        if season_model == 'True':
            model.daily_seasonality = 'auto'
            
            seasonality_mode = st.selectbox('seasonality_mode',
                                        options = ['multiplicative', 'additive'],
                                        index = 1,
                                        help = tooltips_text['seasonality_mode'])
            model.seasonality_mode = seasonality_mode
            
            set_seasonality_prior_scale = st.checkbox('Set seasonality_prior_scale',
                                                      value = True,
                                                      help=tooltips_text['set_overall_seasonality_prior_scale'])
            # add info
            if set_seasonality_prior_scale:
                seasonality_prior_scale = st.number_input('overall_seasonality_prior_scale',
                                                      min_value= 0.01,
                                                      max_value= 50.0,
                                                      value=6.0,
                                                      step = 0.1,
                                                      help = tooltips_text['overall_seasonality_prior_scale'])
            else:
                seasonality_prior_scale = 1.0
                
            model.seasonality_prior_scale = seasonality_prior_scale
            
            yearly_seasonality = st.selectbox('yearly_seasonality', 
                                          ('auto', False, 'Custom'),
                                          help = tooltips_text['add_yearly_seasonality'])
            if yearly_seasonality == 'Custom':
                model.yearly_seasonality = False
                yearly_seasonality_order = st.number_input('Yearly seasonality order',
                                                           min_value = 1,
                                                           max_value=30,
                                                           value=5,
                                                           step=1,
                                                           help = tooltips_text['yearly_order'])
                if set_seasonality_prior_scale is False:
                    yearly_prior_scale = st.number_input('Yearly seasonality prior scale',
                                                           min_value = 0.05,
                                                           max_value=50.0,
                                                           value=6.0,
                                                           step= 0.1,
                                                           help = tooltips_text['yearly_prior_scale'])
                # add yearly seasonality to model
                model.add_seasonality(name='yearly', 
                                      period = 365,
                                      fourier_order = yearly_seasonality_order,
                                      prior_scale = seasonality_prior_scale if set_seasonality_prior_scale else yearly_prior_scale) # add seasonality
            
            monthly_seasonality = st.selectbox('monthly_seasonality', 
                                          options = ('Auto', False, 'Custom'),
                                          index = 2,
                                          help = tooltips_text['add_monthly_seasonality'])
            if monthly_seasonality == 'Custom':
                model.monthly_seasonality = False
                monthly_seasonality_order = st.number_input('Monthly seasonality order',
                                                           min_value = 1,
                                                           max_value=30,
                                                           value=9,
                                                           step=1,
                                                           help = tooltips_text['monthly_order'])
                if set_seasonality_prior_scale is False:
                    monthly_prior_scale = st.number_input('Monthly seasonality prior scale',
                                                           min_value = 0.05,
                                                           max_value=50.0,
                                                           value=6.0,
                                                           step= 0.1,
                                                           help = tooltips_text['monthly_prior_scale'])
                # add monthly seasonality to model
                model.add_seasonality(name='monthly', 
                                      period = 30.4,
                                      fourier_order = monthly_seasonality_order,
                                      prior_scale = seasonality_prior_scale if set_seasonality_prior_scale else monthly_prior_scale) # add seasonality
            
            weekly_seasonality = st.selectbox('weekly_seasonality', 
                                          options = ('Auto', False, 'Custom'),
                                          index = 2,
                                          help = tooltips_text['add_weekly_seasonality'])
            if weekly_seasonality == 'Custom':
                model.weekly_seasonality = False
                weekly_seasonality_order = st.number_input('Weekly seasonality order',
                                                           min_value = 1,
                                                           max_value=30,
                                                           value=6,
                                                           step=1,
                                                           help = tooltips_text['weekly_order'])
                if set_seasonality_prior_scale is False:
                    weekly_prior_scale = st.number_input('Weekly seasonality prior scale',
                                                           min_value = 0.05,
                                                           max_value=50.0,
                                                           value=6.0,
                                                           step= 0.1,
                                                           help = tooltips_text['weekly_prior_scale'])
                # add weekly seasonality to model
                model.add_seasonality(name='weekly', 
                                      period = 7,
                                      fourier_order = weekly_seasonality_order,
                                      prior_scale = seasonality_prior_scale if set_seasonality_prior_scale else weekly_prior_scale) # add seasonality
            
        elif season_model == 'auto':
            # selected 'auto' 
            model.yearly_seasonality = 'auto'
            model.monthly_seasonality = 'auto'
            model.weekly_seasonality = 'auto'
            model.daily_seasonality = 'auto'
        
        else:
            # selected False - no seasonality
            model.yearly_seasonality = False
            model.monthly_seasonality = False
            model.weekly_seasonality = False
            model.daily_seasonality = False
    
    # HOLIDAYS
    # =========================================================================
    with st.sidebar.expander('Holidays'):
        add_holidays = st.checkbox('Add holidays', 
                            value = True,
                            key = 'holiday_model',
                            help = tooltips_text['add_holidays'])
        if add_holidays:
            # add public holidays
            add_public_holidays = st.checkbox('Public holidays',
                                              value = True,
                                              help = tooltips_text['add_public_holidays'])
            if add_public_holidays:
                model.add_country_holidays(country_name='PH')
            
            # add set_holidays
            add_set_holidays = st.checkbox('Saved holidays',
                                           value = True,
                                           help = tooltips_text['add_saved_holidays'])
            if add_set_holidays:
                fathers_day = pd.DataFrame({
                    'holiday': 'fathers_day',
                    'ds': pd.to_datetime(['2022-06-19']),
                    'lower_window': -21,
                    'upper_window': 3})
                
                holidays_set = {'fathers_day': fathers_day}
                
                selected_holidays = st.multiselect('Select saved holidays',
                                                   options=list(holidays_set.keys()),
                                                   default = list(holidays_set.keys()))
                
                model.holidays = pd.concat([holidays_set[h] for h in selected_holidays])
            
            holiday_scale_dict = {}
            
            holiday_scale = st.number_input('holiday_prior_scale',
                                            min_value = 1.0,
                                            max_value = 30.0,
                                            value = float(3),
                                            step = 1.0,
                                            help = tooltips_text['holiday_prior_scale'])
            # set holiday prior scale
            model.holiday_prior_scale = holiday_scale
            
        else:
            # no holiday effects
            model.holidays = None
            model.holiday_prior_scale = 0
    
    # REGRESSORS
    # =========================================================================
    with st.sidebar.expander('Regressors'):
        
        def is_saturday(ds):
            # check if saturday
            date = pd.to_datetime(ds)
            return ((date.dayofweek + 1) == 6)*1
        
        def is_sunday(ds):
            # check if sunday
            date = pd.to_datetime(ds)
            return ((date.dayofweek + 1) == 7)*1
        
        @st.experimental_memo
        def get_gtrend_data(kw_list, df):
            '''
            Get google trend data for specifie keywords
            '''
            pytrend = TrendReq()
            start = pd.to_datetime(df.ds.min())
            end = pd.to_datetime(df.ds.max())
            historicaldf = pytrend.get_historical_interest(kw_list, 
                                    year_start=start.year, 
                                    month_start=start.month, 
                                    day_start=start.day, 
                                    year_end=end.year, 
                                    month_end=end.month, 
                                    day_end=end.day, 
                                    cat=0, 
                                    geo='', 
                                    gprop='', 
                                    sleep=0)
            historicaldf_grp = historicaldf[kw_list].groupby(historicaldf.index.date).mean()
            return historicaldf_grp.fillna(0).asfreq('1D').reset_index()
        
        # add data metrics option
        add_metrics = st.checkbox('Add data metrics',
                                  value = True,
                                  help = tooltips_text['add_metrics'])
        
        exog_num_cols = {'total_qty': ['sessions', 'ad_costs_total', 'ctr_total']}
        
        regressors = list()
        metrics_container = st.empty()
        if add_metrics:
            # Select traffic metrics available from data.
            
            with metrics_container.container():
                traffic_exogs = st.multiselect('Select traffic data metrics',
                               options = exog_num_cols[param],
                               default = exog_num_cols[param],
                               help = tooltips_text['add_metrics_select'])
                
                regressors.extend(traffic_exogs)
                
                
                for traffic_exog in traffic_exogs:
                    evals.loc[:, traffic_exog] = df_traffic[df_traffic.date.isin(evals.ds)][traffic_exog].values
                    model.add_regressor(traffic_exog)
                    
                    # if forecast future
                    if make_forecast_future:
                        future.loc[future.ds.isin(evals.ds), traffic_exog] = df_traffic[df_traffic.date.isin(evals.ds)][traffic_exog].values
                
                
                txns_exogs = st.multiselect('Select transaction data metrics',
                               options = list(sku_dict[group_selected].columns.drop(labels = ['date', 'total_qty'])),
                               default = list(sku_dict[group_selected].columns.drop(labels = ['date', 'total_qty'])),
                               help = tooltips_text['add_metrics_select'])
                
                regressors.extend(txns_exogs)
                
                for txns_exog in txns_exogs:
                    evals.loc[:, txns_exog] = sku_dict[group_selected][sku_dict[group_selected].date.isin(evals.ds)][txns_exog].values
                    model.add_regressor(txns_exog)
                
                    
                    # if forecast future
                    if make_forecast_future:
                        future.loc[future.ds.isin(evals.ds), txns_exog] = sku_dict[group_selected][sku_dict[group_selected].date.isin(evals.ds)][txns_exog].values
        
        # gtrends
        add_gtrends = st.checkbox('Add Google trends',
                              value = False,
                              help = tooltips_text['add_google_trends'])

        gtrends_container = st.empty()
        if add_gtrends:
            # keywords
            with gtrends_container.container():
                kw_list = ['gulong.ph', 'gogulong']
                gtrends_st = st.text_area('Enter google trends keywords',
                                            value = ', '.join(kw_list),
                                            help = tooltips_text['gtrend_kw'])
                # selected keywords
                kw_list = [kw.strip() for kw in gtrends_st.split(',')]
                # cannot generate data for dates in forecast horizon
                gtrends = get_gtrend_data(kw_list, evals)
                for g, gtrend in enumerate(gtrends.columns[1:]):
                    evals.loc[:,gtrend] = gtrends[gtrends['index'].isin(evals.ds)][gtrend]
                    model.add_regressor(gtrend)
                    regressors.append(gtrend)
                    # only update future df if make_future_forecast
                    if make_forecast_future:
                        future.loc[future.ds.isin(evals.ds), gtrend] = gtrends[gtrends['index'].isin(evals.ds)][gtrend]
                    
        
        if make_forecast_future:
            # input regressor data for future/forecast dates
            # if selected metrics is not None
            regressor_input = st.empty()
            if add_metrics and len(regressors) > 0:
                # provide input field
                with regressor_input.container():
                    for regressor in regressors:
                        if regressor in df_traffic.columns:
                            exog_data = df_traffic[df_traffic.date.isin(date_series.ds.values)][regressor]
                        elif regressor in sku_dict[group_selected].columns:
                            exog_data = sku_dict[group_selected][sku_dict[group_selected].date.isin(evals.ds)][regressor]
                        if add_gtrends and len(kw_list) > 0:
                            exog_data = gtrends[regressor]
                        # added key to solve DuplicateWidgetID
                        data_input = st.selectbox(regressor + ' data input type:',
                                             options=['total', 'average'],
                                             index=1,
                                             key = regressor + '_input',
                                             help = tooltips_text['data_input_type'])
                        
                        if data_input == 'total':
                            # if data input is total
                            total = st.number_input('Select {} total over forecast period'.format(regressor),
                                                   min_value = 0.0, 
                                                   value = float(np.nansum(exog_data[-int(forecast_horizon):])),
                                                   step = 0.01,
                                                   help = tooltips_text['data_input_total'])
                            future.loc[future.index[-int(forecast_horizon):],regressor] = np.full((int(forecast_horizon),), round(total/forecast_horizon, 3))
                        else:
                            # if data input is average
                            average = st.number_input('Select {} average over forecast period'.format(regressor),
                                                   min_value = 0.00, 
                                                   value = np.nanmean(exog_data[-int(forecast_horizon):]),
                                                   step = 0.010,
                                                   help = tooltips_text['data_input_average'])
                            future.loc[future.index[-int(forecast_horizon):],regressor] = np.full((int(forecast_horizon),), round(average, 3))
            else:
                # delete unused fields
                regressor_input.empty()
            
            
            
            # custom regressors (functions applied to dates)
            add_custom_reg = st.checkbox('Add custom regressors',
                                         value = True,
                                         help = tooltips_text['add_custom_regressors'])
            
            custom_reg_container = st.empty()
            if add_custom_reg:
                
                with custom_reg_container.container():
                    regs = {'is_saturday': evals.ds.apply(is_saturday),
                        'is_sunday'  : evals.ds.apply(is_sunday)}
                
                    if make_forecast_future:
                        regs_future = {'is_saturday': future.ds.apply(is_saturday),
                                   'is_sunday'  : future.ds.apply(is_sunday)}
                    # regressor multiselect
                    regs_list = st.multiselect('Select custom regs',
                                   options = list(regs.keys()),
                                   default = list(regs.keys()))
                    
                    for reg in regs_list:
                        evals.loc[:, reg] = regs[reg].values
                        model.add_regressor(reg)
                    
                        if make_forecast_future:
                            future.loc[:, reg] = regs_future[reg].values
            else:
                custom_reg_container.empty()
                
    # OUTLIERS
    # =====================================================================
    with st.sidebar.expander('Outliers'):
        remove_outliers = st.checkbox('Remove outliers', value = False,
                                      help = tooltips_text['outliers'])
        if remove_outliers:
            # option to remove datapoints with value = 0
            remove_neg = st.checkbox('Remove negative datapoints', 
                                       value = True)
            if remove_neg:
                evals = evals.apply(lambda x: x['y'] if x['y'] > 0 else 0, axis=1)
    
    start_forecast = st.sidebar.checkbox('Launch forecast',
                                 value = False)     
    
    
    if len(regressors) > 0:
        st.subheader('Regressor preview')
        selected_reg = st.multiselect('Select regressors to show',
                                      options = regressors,
                                      default = regressors)
        # plotly scatter plot of regressors
        plot_regressors(evals, selected_reg)
    
    if start_forecast:
        # show dataframe for debugging
        # st.dataframe(future)
        model.fit(evals)
        if make_forecast_future:
            #st.dataframe(future)
            forecast = model.predict(future)
        else:
            #st.dataframe(evals)
            forecast = model.predict(evals)
        
        if remove_neg:
            forecast = forecast.apply(lambda x: remove_neg_val(x['yhat'], x['yhat_lower'], x['yhat_upper']), axis=1)
        else:
            pass


        # plot
        st.header('Overview')
        st.markdown(tooltips_text['overview'])
        st.plotly_chart(plot_plotly(model, forecast,
                                    uncertainty=True,
                                    changepoints=True,
                                    ylabel = param,
                                    xlabel = 'date',
                                    figsize=(800, 600)))
        
        if make_forecast_future:
            # get forecasted values
            df_preds = forecast.tail(forecast_horizon)
            df_preds.loc[:, 'ds'] = pd.to_datetime(df_preds.loc[:, 'ds'], unit='D').dt.strftime('%Y-%m-%d')
            df_preds = df_preds.set_index('ds')
            
            # display results
            st.subheader('Forecast results')
            fcast_col1, fcast_col2 = st.columns([2, 1])
            with fcast_col1:
                
                cols = ['yhat', 'yhat_lower', 'yhat_upper']
                # cols.extend(regressors)
                st.dataframe(df_preds[cols])
                
                st.download_button(label='Export forecast results',
                                   data = convert_csv(df_preds[['yhat', 'yhat_lower', 'yhat_upper']]),
                                   file_name = param +'_forecast_results.csv')
                
            with fcast_col2:    
                view_setting = st.selectbox('View sum or mean',
                             options=['sum', 'mean'],
                             index = 0)
                if view_setting =='sum':
                    st.dataframe(df_preds[['yhat', 'yhat_lower', 'yhat_upper']].sum().rename('total_' + param))
                elif view_setting == 'mean':    
                    st.dataframe(df_preds[['yhat', 'yhat_lower', 'yhat_upper']].mean().rename('average_' + param))
            
                
                
            
        #st.expander('Plot info'):
        st.header('Evaluation and Error analysis')
        
        st.subheader('Global performance')
        mae = round(mean_absolute_error(evals.y, forecast.loc[evals.index,'yhat']), 3)
        mape = round(mean_absolute_percentage_error(evals.y, forecast.loc[evals.index,'yhat'])*100, 3)
        rmse = round(np.sqrt(mean_squared_error(evals.y, forecast.loc[evals.index,'yhat'])), 3)
        
        err1, err2, err3 = st.columns(3)
        with err1:
            st.metric('MAE', 
                      value = mae, 
                      help = tooltips_text['mae'])
        
        with err2:
            st.metric('RMSE', 
                      value = rmse, 
                      help = tooltips_text['rmse'])
        
        with err3:
            st.metric('MAPE', 
                      value = round(mape, 3), 
                      help = tooltips_text['mape'])
            
        st.subheader('Forecast vs Actual')
        truth_vs_forecast = plot_forecast_vs_actual_scatter(evals, forecast)
        st.plotly_chart(truth_vs_forecast)
        st.markdown(tooltips_text['forecast_vs_actual'])
        r2 = round(r2_score(evals.y, forecast.loc[evals.index,'yhat']), 3)
        #st.markdown('**<p style="font-size: 20px">R<sup>2</sup>** </p>', unsafe_allow_html = True)
        st.metric(label = 'Pearson Coefficient',
                  value = r2,
                  help = tooltips_text['pearson_coeff'])
        #with st.expander('Pearson correlation coefficient'):
        #    st.markdown(tooltips_text['pearson_coeff'], unsafe_allow_html = True)
        
        
        st.header('Impact of components')
        st.markdown(tooltips_text['impact_of_components'])
        st.plotly_chart(plot_components_plotly(
            model,
            forecast,
            uncertainty=True))
        
        if len(regressors) > 0:
            st.dataframe(regressor_coefficients(model))
   