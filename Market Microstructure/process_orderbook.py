import os
import shutil
import glob
import time
from multiprocessing import Pool
import datetime
import argparse

import pandas as pd
import numpy as np

def args_acquisition():

	"""
	A method to parse arguements useful for the whole execution.

	Returns:
		args: Default/User defined arguements useful for the whole execution.
	"""

	parser = argparse.ArgumentParser(description='Argument parser for LOBSTER data preliminary processing.')
	parser.add_argument('--stock_name', type=str, default='GOOG', help="Stock's name setup.")
	parser.add_argument('--starting_time', type=int, default=9, help='Starting time. Trades before this hour are excluded.')
	parser.add_argument('--ending_time', type=int, default=16, help='Ending time. Trades after this hour are excluded.')
	args = parser.parse_args()

	return args

def get_root_path(args):

	"""
	A method to retrieve the root path for experiments. A root path is composed by the stock name (i.e. ./IEX).

	Args:
		args (argparse.Namespace): All argument passed by the user and/or default ones.

	Returns:
		root_path_string: String containing the root path.
	"""

	root_path_string = str("./Stocks/" + args.stock_name)

	return root_path_string

def delete_paths(list_paths_to_be_deleted):

	"""
	A method to delete specific paths.

	Args:
		list_paths_to_be_deleted (list): List containing all paths to be deleted.
	"""

	for path in list_paths_to_be_deleted:

		if os.path.exists(path):
			shutil.rmtree(path, ignore_errors=True)
			

def create_paths(list_paths_to_be_created):

	"""
	A method to create specific paths.

	Args:
		list_paths_to_be_created (list): List containing all paths to be created.
	"""

	for path in list_paths_to_be_created:

		try:
			os.makedirs(path, exist_ok=True)
		except OSError:
			print ("Creation of the directory %s failed" % path)

def split_path_name(file_name, part):

	"""
	A method to break paths into their two main components: head and tail.

	Args:
		file_name (string): Path to be broken into different components.
		part (string): Component to be considered (i.e. head/tail).

	Returns:
		head_tail_split: Path's component to be considered.
	"""

	head_tail_split = os.path.split(file_name)

	if part == 'head':
		return head_tail_split[0] 
	else:
		return head_tail_split[1]

def organize_o_m_pairs(orderbook_file_names, message_file_names, args):

	"""
	A method to create a list containing couples in the form <orderbook_file_name, message_file_name>.

	Args:
		orderbook_file_names (list): List containing all orderbook files' names.
		message_file_names (list): List containing all message files' names.
		args (argparse.Namespace): All argument passed by the user and/or default ones.

	Returns:
		file_data: List containing couples in the form <orderbook_file_name, message_file_name>.
	"""

	files_data = []
	for o, m in zip(orderbook_file_names, message_file_names):
		files_data.append((o, m, args))

	return files_data

def rename_orderbook_columns(orderbook):

	"""
	A method to give significant names to LOBSTER orderbook data (columns).

	Args:
		orderbook (string): Path containing the file to be transformed.

	Returns:
		orderbook: Dataframe containing transformed data.
	"""

	orderbook = pd.read_csv(orderbook, header=None, index_col=False)
	list_names = []
	for column in range(0, orderbook.shape[1]):
		
		if column % 4 == 0:
			list_names.append(str('Ask_Price_Level_' + str(int(column/4) + 1)))
			orderbook[column] = orderbook[column].div(10000)
		elif column % 4 == 1:
			list_names.append(str('Ask_Volume_Level_' + str(int(column/4) + 1)))
		elif column % 4 == 2:
			list_names.append(str('Bid_Price_Level_' + str(int(column/4) + 1)))
			orderbook[column] = orderbook[column].div(10000)
		else:
			list_names.append(str('Bid_Volume_Level_' + str(int(column/4) + 1)))

	orderbook.set_axis(list_names, axis=1, inplace=True)
	return orderbook

def rename_message_columns(message):

	"""
	A method to give significant names to LOBSTER message data (columns).

	Args:
		message (string): Path containing the file to be transformed.

	Returns:
		message: Dataframe containing transformed data.
	"""

	name_message_file = os.path.split(message)[-1]
	date = name_message_file.split('_')[1] + " 00:00:00"

	message = pd.read_csv(message, header=None, index_col=False).dropna(axis=1)
	list_names = ['DateTime', 'Event_Type', 'Order_ID', 'Size', 'Price', 'Direction']

	message.set_axis(list_names, axis=1, inplace=True)

	message['DateTime'] = [(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(seconds=d)).strftime("%Y-%m-%d %H:%M:%S:%f") for d in message.DateTime]
	return message

def merge_orderbook(orderbook, message):

	"""
	A method to merge LOBSTER orderbook and message files.

	Args:
		orderbook (pandas.DataFrame): Dataframe corresponding to the orderbook file.
		message (pandas.DataFrame): Dataframe corresponding to the message file.

	Returns:
		merged_orderbook: Dataframe containing merged dataframe.
	"""

	merged_orderbook = pd.concat([message, orderbook], axis=1, join='inner')
	return merged_orderbook

def manage_halts(merged_orderbook):

	"""
	A method to filter out trading halts from LOBSTER files (both orderbook and message files).

	Args:
		merged_orderbook (pandas.DataFrame): Dataframe containing both orderbok and message dataframes in a unique object.

	Returns:
		merged_orderbook: Dataframe filtered from all trading halts signals.
	"""

	return merged_orderbook.drop(merged_orderbook[merged_orderbook.Event_Type == 7].index)

def manage_unreasonable_prices(merged_orderbook):

	"""
	A method to filter out unreasonable prices. A price is considered unreasonable if the best bid price (i.e. bid price at the first LOB level) is higher than the best ask price (i.e. ask price at the first LOB level).

	Args:
		merged_orderbook (pandas.DataFrame): Dataframe containing both orderbok and message dataframes in a unique object.

	Returns:
		merged_orderbook: Dataframe filtered from all unreasonable prices.
	"""
	
	merged_orderbook = merged_orderbook.drop(merged_orderbook[merged_orderbook.Bid_Price_Level_1 > merged_orderbook.Ask_Price_Level_1].index)

	return merged_orderbook

def manage_auctions(merged_orderbook):

	"""
	A method to filter out opening and closing auctions. Opening auctions are identified as messages with 'type == 6' and 'ID == -1' and closing auctions can be identified as messages with 'type == 6' and 'ID == -2'.

	Args:
		merged_orderbook (pandas.DataFrame): Dataframe containing both orderbok and message dataframes in a unique object.

	Returns:
		merged_orderbook: Dataframe filtered from opening and closing auctions.
	"""

	opening_auction_dates = merged_orderbook.loc[(merged_orderbook["Event_Type"] == 6) & (merged_orderbook["Order_ID"] == -1), "DateTime"].values
	closing_auction_dates = merged_orderbook.loc[(merged_orderbook["Event_Type"] == 6) & (merged_orderbook["Order_ID"] == -2), "DateTime"].values

	for oa in opening_auction_dates:
		merged_orderbook = merged_orderbook[~(merged_orderbook['DateTime'] <= oa)]
	for ca in closing_auction_dates:
		merged_orderbook = merged_orderbook[~(merged_orderbook['DateTime'] >= ca)]

	del opening_auction_dates
	del closing_auction_dates

	return merged_orderbook

def manage_unique_timestamps(merged_orderbook):

	"""
	A method to merge transactions with unique timestamp.

	Args:
		merged_orderbook (pandas.DataFrame): Dataframe containing both orderbok and message dataframes in a unique object.

	Returns:
		merged_orderbook: Dataframe where transaction with the same timestamp are merged together.
	"""

	trades = merged_orderbook.iloc[:, 0:6]
	trades = trades.loc[(trades['Event_Type'] == 4) | (trades['Event_Type'] == 5)]
	
	df = merged_orderbook.groupby(by=['DateTime']).first().reset_index()

	df['Midquote'] = df['Ask_Price_Level_1'].div(2) + df['Bid_Price_Level_1'].div(2)
	df['Lag_Midquote'] = df['Midquote'].shift(1)
	df = df[['DateTime', 'Ask_Price_Level_1', 'Bid_Price_Level_1', 'Midquote', 'Lag_Midquote']]

	df = trades.merge(df, on="DateTime", how='inner')

	del trades

	df.loc[((df.Event_Type == 5) & (df.Price < df.Lag_Midquote)), 'Direction'] = 1
	df.loc[((df.Event_Type == 5) & (df.Price > df.Lag_Midquote)), 'Direction'] = -1

	grouped_df = df.groupby(by=['DateTime'])

	del df

	new_df_grouped = {'DateTime':[], 'Event_Type':[], 'Order_ID':[], 'Price':[], 'Size':[], 'Direction':[]}
	
	for name, group in grouped_df:
		new_df_grouped['DateTime'].append(name)
		new_df_grouped['Event_Type'].append(list(group['Event_Type'])[-1])
		new_df_grouped['Order_ID'].append(None)
		new_df_grouped['Price'].append(np.sum(group['Price'] * group['Size']) / np.sum(group['Size']))
		new_df_grouped['Size'].append(np.sum(group['Size']))
		new_df_grouped['Direction'].append(list(group['Direction'])[-1])

	new_df_grouped = pd.DataFrame(new_df_grouped)
	
	df_2 = merged_orderbook.iloc[:, 6:]
	df_2['DateTime'] = merged_orderbook['DateTime']
	df_2 = df_2.groupby(by=['DateTime']).last().reset_index()
	new_df_grouped = new_df_grouped.merge(df_2, on='DateTime', how='inner')

	filtered_df = merged_orderbook.loc[(merged_orderbook['Event_Type'] != 4) & (merged_orderbook['Event_Type'] != 5)]
	
	merged_orderbook = pd.concat([filtered_df, new_df_grouped], axis=0, sort=False).sort_values(by='DateTime')

	del filtered_df, new_df_grouped

	return merged_orderbook

def save_file(orderbook_file_name, merged_orderbook, args):

	"""
	A method to save cleaned and merged orderbook/message files.

	Args:
		orderbook_file_name (string): Name of the original orderbook (and, consequently, message file).
		merged_orderbook (pandas.DataFrame): Dataframe containing both orderbok and message dataframes in a unique object.
		args (argparse.Namespace): All argument passed by the user and/or default ones.
	"""

	tail = split_path_name(orderbook_file_name, 'tail')

	merged_orderbook['Time'] = pd.to_datetime(merged_orderbook['DateTime'], format='%Y-%m-%d %H:%M:%S:%f').dt.time
	merged_orderbook = merged_orderbook[merged_orderbook['Time'] > datetime.time(args.starting_time)] 
	merged_orderbook = merged_orderbook[merged_orderbook['Time'] < datetime.time(args.ending_time)]
	merged_orderbook.drop(['Time'], axis=1, inplace=True)

	merged_orderbook.to_csv(str(get_root_path(args) + "/cleaned_data/" + tail), index=False)

def execution_time_extimation(function, *args):

	"""
	A method to perform a sanity check on each function's execution time. This method has been introduced because treated files are big-data files.

	Args:
		function (string): Name of the function to be executed.
		*args (argparse.Namespace): Function's parameters.

	Returns:
		returning_object: Object returned by the function's execution.
		time_returned: Time requested to execute the function.
	"""

	tic = time.perf_counter()
	returning_object = function(*args)
	toc = time.perf_counter()

	time_returned = toc - tic
	return returning_object, time_returned


def filter_orderbook(orderbook, message, args):

	"""
	A method to execute all the LOBSTER data filtering pipeline.

	Args:
		orderbook (string): Path of the orderbook file.
		message (string): Path of the message file.
		args (argparse.Namespace): All argument passed by the user and/or default ones.
	"""

	orderbook_file_name = orderbook
	message_file_name = message

	orderbook, time_returned = execution_time_extimation(rename_orderbook_columns, orderbook)

	message, time_returned = execution_time_extimation(rename_message_columns, message)

	merged_orderbook, time_returned = execution_time_extimation(merge_orderbook, orderbook, message)
	
	del orderbook, message

	merged_orderbook, time_returned = execution_time_extimation(manage_halts, merged_orderbook)

	merged_orderbook, time_returned = execution_time_extimation(manage_unreasonable_prices, merged_orderbook)
	
	merged_orderbook, time_returned = execution_time_extimation(manage_auctions, merged_orderbook)
	
	merged_orderbook, time_returned = execution_time_extimation(manage_unique_timestamps, merged_orderbook)

	save_file(orderbook_file_name, merged_orderbook, args)

if __name__ == '__main__':

	args = args_acquisition()

	path_raw_data = str(get_root_path(args) + "/raw_data")
	path_cleaned_data = str(get_root_path(args) + "/cleaned_data")

	list_paths_to_be_created = [path_cleaned_data]
	delete_paths(list_paths_to_be_created)
	create_paths(list_paths_to_be_created)

	orderbook_file_names = sorted(glob.glob(str(path_raw_data + '/*_orderbook_10.csv')))
	message_file_names = sorted(glob.glob(str(path_raw_data + '/*_message_10.csv')))

	files_data = organize_o_m_pairs(orderbook_file_names, message_file_names, args)

	process_pool = Pool(4)
	process_pool.starmap(filter_orderbook, files_data)

	#filter_orderbook(orderbook_file_names, message_file_names)