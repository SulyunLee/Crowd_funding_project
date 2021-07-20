
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import nltk
from textblob import TextBlob
from datetime import date
from datetime import timedelta

def compute_actual_comment_timestamp(comments_df, file_saved_time):
    # find the actual timestamps of the comments
    comment_actual_timestamp = np.zeros(comments_df.shape[0], dtype='datetime64[ns]')
    for idx, row in tqdm(comments_df.iterrows(), total=comments_df.shape[0]):
        timestamp = row['timestamp']
        
        if 'minutes ago' in timestamp:
            comment_actual_timestamp[idx] = np.datetime64(file_saved_time)
        elif 'hours ago' in timestamp:
            comment_actual_timestamp[idx] = np.datetime64(file_saved_time)
        elif 'days ago' in timestamp:
            delta_time = int([n for n in timestamp.split() if n.isnumeric()==True][0])
            comment_time = np.datetime64(file_saved_time - timedelta(days=delta_time))
            comment_actual_timestamp[idx] = comment_time
        elif 'months ago' in timestamp:
            delta_time = int([n for n in timestamp.split() if n.isnumeric()==True][0])
            comment_time = np.datetime64(file_saved_time - timedelta(days=delta_time*30))
            comment_actual_timestamp[idx] = comment_time
        elif 'years ago' in timestamp:
            delta_time = int([n for n in timestamp.split() if n.isnumeric()==True][0])
            comment_time = np.datetime64(file_saved_time - timedelta(days=delta_time*365))
            comment_actual_timestamp[idx] = comment_time
        elif '1 hour ago' in timestamp:
            comment_actual_timestamp[idx] = np.datetime64(file_saved_time)
        elif '1 day ago' in timestamp:
            comment_time = np.datetime64(file_saved_time - timedelta(days=1))
            comment_actual_timestamp[idx] = comment_time
        elif '1 month ago' in timestamp:
            comment_time = np.datetime64(file_saved_time - timedelta(days=30))
            comment_actual_timestamp[idx] = comment_time
        elif '1 year ago' in timestamp:
            comment_time = np.datetime64(file_saved_time - timedelta(days=365))
            comment_actual_timestamp[idx] = comment_time

    comments_df['actual_timestamp'] = comment_actual_timestamp
    return comments_df




if __name__ == "__main__":
    updates_file = "temp_data/newproject_rf_predictions_ntree128_md32_delayonly.csv"
    comments_file = "../dataset/indiegogo_comments_newprojects.csv"
    additional_comments_file = "temp_data/comments_around_delays.pickle"


    # infile = open(additional_comments_file, 'rb')
    # additional_comments_df = pickle.load(infile)
    # infile.close()

    updates_df = pd.read_csv(updates_file)
    comments_df = pd.read_csv(comments_file)

    updates_df['date'] = updates_df['date'].astype('datetime64[ns]')

    # Compute the sentiment scores for comments and additional comments
    print('Computing the comment sentiments...')
    comment_scores = np.zeros(comments_df.shape[0]) 
    for idx, row in tqdm(comments_df.iterrows(), total=comments_df.shape[0]):
        comment = row['content_text']
        try:
            textblob_obj = TextBlob(comment)
            sentiment = textblob_obj.sentiment.polarity
            comment_scores[idx] = sentiment
        except:
            comment_scores[idx] = 0
    comments_df['sent_score'] = comment_scores

    # additional_comment_scores = np.zeros(additional_comments_df.shape[0])
    # for idx, row in tqdm(additional_comments_df.iterrows(), total=additional_comments_df.shape[0]):
        # comment = row['content']
        # try:
            # textblob_obj = TextBlob(comment)
            # sentiment = textblob_obj.sentiment.polarity
            # additional_comment_scores[idx] = sentiment
        # except:
            # additional_comment_scores[idx] = 0
    # additional_comments_df['sent_score'] = additional_comment_scores

    # compute the actual timestamps for comments
    print('Computing the actual timestamps...')
    comments_df = compute_actual_comment_timestamp(comments_df, date(2020, 6, 21))
    # additional_comments_df = compute_actual_comment_timestamp(additional_comments_df, date(2020, 2, 24))

    # find the comments written until 1 year after the updates
    comment_avg_sent_score = np.zeros(updates_df.shape[0])
    before_comment_avg_sent_score = np.zeros(updates_df.shape[0])
    print('Computing the average comment scores for each updates...')
    for idx, row in updates_df.iterrows():
        update_timestamp = row['date']
        projectid = row['projectid']

        # find the comments that match with the project id
        update_comments1 = comments_df[comments_df['projectID'] == projectid]
        # update_comments2 = additional_comments_df[additional_comments_df['project_id'] == projectid]

        # compute the 1 year after timestamp
        timestamp_after_1yr = update_timestamp + timedelta(days=365)
        timestamp_before_1yr = update_timestamp - timedelta(days=365)


        # find the comments from dataframes
        extracted_comments1 = update_comments1[(update_comments1['actual_timestamp'] >= update_timestamp) & (update_comments1['actual_timestamp'] < timestamp_after_1yr)]
        # extracted_comments2 = update_comments2[(update_comments2['actual_timestamp'] >= update_timestamp) & (update_comments2['actual_timestamp'] < timestamp_after_1yr)]

        before_comments1 = update_comments1[(update_comments1['actual_timestamp'] >= timestamp_before_1yr) &
                                            (update_comments1['actual_timestamp'] < update_timestamp)]
        # before_comments2 = update_comments2[(update_comments2['actual_timestamp'] > timestamp_before_1yr) &
                                            # (update_comments2['actual_timestamp'] < update_timestamp)]

        # concatenate these extracted comments from two dataframes
        # extracted_comments_combined = pd.concat([extracted_comments1, extracted_comments2], sort=True)
        # before_comments_combined = pd.concat([before_comments1, before_comments2], sort=True)

        # compute the average sentiment scores
        # avg_sent_score = extracted_comments_combined.sent_score.mean()
        avg_sent_score = extracted_comments1.sent_score.mean()
        comment_avg_sent_score[idx] = avg_sent_score

        # avg_before_sent_score = before_comments_combined.sent_score.mean()
        avg_before_sent_score = before_comments1.sent_score.mean()
        before_comment_avg_sent_score[idx] = avg_before_sent_score
        

    updates_df['comment_sent_score'] = comment_avg_sent_score
    updates_df['before_comment_sent_score'] = before_comment_avg_sent_score
    # updates_df['proj_success'] = [1 if x > 0 else 0 for x in updates_df['comment_sent_score']]
    updates_df.to_csv("temp_data/newproject_rf_predictions_ntree128_md32_delay_response_success.csv", index=False)




