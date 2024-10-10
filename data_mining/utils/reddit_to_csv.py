# this converts a zst file to csv
#
# it's important to note that the resulting file will likely be quite large
# and you probably won't be able to open it in excel or another csv reader
#
# arguments are inputfile, outputfile, fields
# call this like
# python to_csv.py wallstreetbets_submissions.zst wallstreetbets_submissions.csv author,selftext,title

import zstandard
import os
import json
from argparse import ArgumentParser
import csv
from datetime import datetime as dt
import logging.handlers
from config import *
import pandas as pd

# put the path to the input file
input_file_path = r"\\MYCLOUDPR4100\Public\reddit\subreddits\intel_comments.zst"
# put the path to the output file, with the csv extension
output_file_path = r"\\MYCLOUDPR4100\Public\intel_comments.csv"
# if you want a custom set of fields, put them in the following list. If you leave it empty the script will use a default set of fields
fields = []

log = logging.getLogger("bot")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
	chunk = reader.read(chunk_size)
	bytes_read += chunk_size
	if previous_chunk is not None:
		chunk = previous_chunk + chunk
	try:
		return chunk.decode()
	except UnicodeDecodeError:
		if bytes_read > max_window_size:
			raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
		return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def read_lines_zst(file_name):
	with open(file_name, 'rb') as file_handle:
		buffer = ''
		reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
		while True:
			chunk = read_and_decode(reader, 2**27, (2**29) * 2)
			if not chunk:
				break
			lines = (buffer + chunk).split("\n")

			for line in lines[:-1]:
				yield line, file_handle.tell()

			buffer = lines[-1]
		reader.close()

def clear_chunk(file_name, chunk):
	if file_name.startswith('RS'):
		print(chunk.columns)
		# Check if 'title' and 'text' columns has any missing values, if so, delete those rows
		chunk.dropna(subset=['title', 'text'], inplace=True)

		# Delete rows where the 'title' column has the value '[removed]'
		chunk = chunk[chunk['title'] != '[removed]']

		# Delete rows where the 'text' column has the value '[deleted]' 
		chunk = chunk[chunk['text'] != '[deleted]']

		# Delete rows where the 'text' column has the value '[removed]'
		chunk = chunk[chunk['text'] != '[removed]']

		# Check if 'title' or 'text' contains the word 'bitcoin' or 'btc' not case-sensitive
		chunk = chunk[chunk['title'].str.contains('bitcoin|btc', case=False) | chunk['text'].str.contains('bitcoin|btc', case=False)]

		# Extract the code from the 'link' column
		chunk['code'] = chunk['link'].str.extract(r'comments/(\w+)/')
	if file_name.startswith('RC'):
		print(chunk.columns)
		# Check if 'title' and 'text' columns has any missing values, if so, delete those rows
		chunk.dropna(subset=['body'], inplace=True)

		# Delete rows where the 'text' column has the value '[removed]'
		chunk = chunk[chunk['body'] != '[removed]']

		# Delete rows where the 'text' column has the value '[deleted]'
		chunk = chunk[chunk['body'] != '[deleted]']

		# Extract the code from the 'link' column
		chunk['code'] = chunk['link'].str.extract(r'comments/(\w+)/')

	return chunk

def check_corrispondence_between_submissions_and_comments(submissions, chunk):
	# Check if the 'code' column in the comments DataFrame is in the 'code' column of the submissions DataFrame
	chunk = chunk[chunk['code'].isin(submissions['code'])]

	return chunk

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--path", type=str)

	args = parser.parse_args()
	assert args.path is not None, "You must specify an input path"

	print(f"Parsed arguments are {args}")
	print(f"Vars parsed arguments are {vars(args)}")

    # Parse the arguments
	path = args.path
	
	# Load file to convert and clean
	INPUT_PATH = f"D:/Downloads/reddit/{path}/"
	OUTPUT_PATH = f"D:/Downloads/reddit/{path}/"

	## CONVERTING
	#For each file in the folder
	for file in os.listdir(INPUT_PATH):
		# Check if it is a zst file
		if file.endswith(".zst"):
			# Get the name of the file
			name = file.split(".")[0]
			# Create the input and output file paths
			input_file_path = f"{INPUT_PATH}{file}"
			output_file_path = f"{OUTPUT_PATH}{name}.csv" 

			# Check if the file is a submission file
			is_submission = "submission" in input_file_path
			if not len(fields):
				if is_submission:
					fields = ["author","title","score","created","link","text","url"]
				else:
					fields = ["author","score","created","link","body"]
					
			# Get the size of the file
			file_size = os.stat(input_file_path).st_size
			file_lines, bad_lines = 0, 0
			line, created = None, None
			output_file = open(output_file_path, "w", encoding='utf-8', newline="")
			writer = csv.writer(output_file)
			writer.writerow(fields)
			try:
				for line, file_bytes_processed in read_lines_zst(input_file_path):
					try:
						obj = json.loads(line)
						output_obj = []
						for field in fields:
							if field == "created":
								value = dt.fromtimestamp(int(obj['created_utc'])).strftime("%Y-%m-%d %H:%M")
							elif field == "link":
								if 'permalink' in obj:
									value = f"https://www.reddit.com{obj['permalink']}"
								else:
									value = f"https://www.reddit.com/r/{obj['subreddit']}/comments/{obj['link_id'][3:]}/_/{obj['id']}/"
							elif field == "author":
								value = f"u/{obj['author']}"
							elif field == "text":
								if 'selftext' in obj:
									value = obj['selftext']
								else:
									value = ""
							else:
								value = obj[field]

							output_obj.append(str(value).encode("utf-8", errors='replace').decode())
						writer.writerow(output_obj)

						created = dt.utcfromtimestamp(int(obj['created_utc']))
					except json.JSONDecodeError as err:
						bad_lines += 1
					file_lines += 1
					if file_lines % 100000 == 0:
						log.info(f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {file_lines:,} : {bad_lines:,} : {(file_bytes_processed / file_size) * 100:.0f}%")
			except KeyError as err:
				log.info(f"Object has no key: {err}")
				log.info(line)
			except Exception as err:
				log.info(err)
				log.info(line)
			
			output_file.close()
			log.info(f"Complete : {file_lines:,} : {bad_lines:,}")
 
			## CLEANING

			# Load submissions_cleaned file
			PATH_SUBMISSIONS = f"D:/Downloads/reddit/submissions/RS_2024-07_cleaned.csv"
			# Load only the 'code' column
			submissions = pd.read_csv(PATH_SUBMISSIONS, usecols=['code'])
   
			log.info(f'Cleaning {file}...')
			import pandas as pd
			from tqdm import tqdm

			input_file_path = output_file_path
			output_file_path = f"{OUTPUT_PATH}{name}_cleaned.csv" 

			chunk_size = 1000000

			# Create a DataFrame to store the cleaned data
			chunk = pd.DataFrame()

			# Save the cleaned DataFrame to a new CSV file
			chunk.to_csv(output_file_path, index=False)

			for chunk in tqdm(pd.read_csv(input_file_path, chunksize=chunk_size)):
				# Iterate over the file in chunks
				chunk = clear_chunk(file, chunk)

				# Check if the 'code' column in the comments DataFrame is in the 'code' column of the submissions DataFrame
				chunk = check_corrispondence_between_submissions_and_comments(submissions, chunk)

				# Turn chunk into a DataFrame
				chunk = pd.DataFrame(chunk)

				# Append the chunk to the .csv file
				chunk.to_csv(output_file_path, mode='a', header=True, index=False)

			log.info(f'Cleaning {file} complete.')