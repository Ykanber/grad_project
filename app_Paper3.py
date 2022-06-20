from Levenshtein import distance as lev
from datetime import *
import pandas as pd
import numpy as np
#Import Flask modules
from flask import Flask, request, render_template
import statistics

#Import pickle to save our model
import pickle 

#twitter libraries
import tweepy
import config
    
def followersToFriendsRatio(followers_count, friends_count):
    if followers_count == 0 or friends_count == 0:
        return 0
    else:
        return followers_count / friends_count

def reputationScore(followers_count, friends_count):
    if followers_count == 0 or friends_count == 0:
        return 0
    else:
        return followers_count / (followers_count + friends_count)

auth = tweepy.OAuth1UserHandler(
   config.API_KEY, config.API_SECRET,
   config.ACCESS_TOKEN, config.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

#Open our model 
model = pickle.load(open('myModel.pkl','rb')) #MODEL IS HERE
#model = pickle.load(open('myModelXGB.pkl','rb')) #MODEL IS HERE
#model = pickle.load(open('myModelADA.pkl','rb')) #MODEL IS HERE

#Initialize Flask and set the template folder to "template"
app = Flask(__name__, template_folder = 'template')

#create our "home" route using the "index.html" page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/FAQ', methods=["GET", "POST"])
def FAQ():
    return render_template('FAQ.html')

@app.route('/privacy', methods=["GET", "POST"])
def privacy():
    return render_template('privacy.html')

#Set a post method to yield predictions on page
@app.route('/', methods = ['POST'])
def predict():
    
    #obtain all form values and place them in an array, convert into integers
    Account_name = request.form['account_name']
    #predict the price given the values inputted by user

    try:
        user = api.get_user(screen_name = Account_name)
    except tweepy.errors.Forbidden:
        return render_template('index.html', prediction_text = "User " + Account_name + " is suspended.")
    except tweepy.errors.NotFound:
        return render_template('index.html', prediction_text = "User " + Account_name + " doesn't exist.")

    tweets = api.user_timeline(screen_name = Account_name, count=100)

    def extractTweetFeatures(traditionalFinalFeatures):
        totalTweetsCount = 0
        totalTweetSize = 0
        totalHashtagCount = 0
        totalRetweetCount = 0
        totalUrlCount = 0
        totalMentionCount = 0
        totalRetweetToTweetsCount = 0
        totalLikesToTweetsCount = 0
        totalTimeBetweenTweets = 0
        TTISD = 0
        totalHoursOfTweets = 0
        TTSD = 0
        tweetTimesArray = []  # Time Format: '2022-05-26 16:25:52+00:00'
        tweetTimeIntervalsArray = []
        tweetHoursInDay = []
        for tweet in tweets:
            totalTweetsCount += 1
            tweetText = tweet.text
            totalTweetSize += len(tweetText)
            totalHashtagCount += len(tweet.entities["hashtags"])
            totalUrlCount += len(tweet.entities["urls"])
            totalMentionCount += len(tweet.entities["user_mentions"])
            totalRetweetToTweetsCount += tweet.retweet_count
            totalLikesToTweetsCount += tweet.favorite_count
            if tweetText[0:4] == 'RT @':
                totalRetweetCount += 1
            tweetTimesArray.append(tweet.created_at)

        if len(tweetTimesArray) >= 4:
            firstTweetTime = tweetTimesArray[0]
            totalHoursOfTweets += firstTweetTime.hour
            tweetHoursInDay.append(firstTweetTime.hour)
            for i in range(1, len(tweetTimesArray)):
                traverseTweetTime = tweetTimesArray[i]
                totalHoursOfTweets += traverseTweetTime.hour
                tweetHoursInDay.append(traverseTweetTime.hour)
                timeDifferenceBetweenTweets = firstTweetTime - traverseTweetTime
                tweetTimeIntervalsArray.append(timeDifferenceBetweenTweets.total_seconds() / 86400)
                totalTimeBetweenTweets += timeDifferenceBetweenTweets.total_seconds() / 86400
                firstTweetTime = traverseTweetTime
            TTISD = statistics.stdev(tweetTimeIntervalsArray)
            TTSD = statistics.stdev(tweetHoursInDay)

        traditionalFinalFeatures['TTIM'] = 0 if totalTweetsCount == 0 else totalTimeBetweenTweets / totalTweetsCount
        traditionalFinalFeatures['TTISD'] = TTISD
        traditionalFinalFeatures['TTM'] = 0 if totalTweetsCount == 0 else totalHoursOfTweets / totalTweetsCount
        traditionalFinalFeatures['TTSD'] = TTSD
        traditionalFinalFeatures['avg_tweet_size'] = 0 if totalTweetsCount == 0 else totalTweetSize / totalTweetsCount
        traditionalFinalFeatures['avg_hashtag_count'] = 0 if totalTweetsCount == 0 else totalHashtagCount / totalTweetsCount
        traditionalFinalFeatures['retweeted_to_total_tweets'] = 0 if totalTweetsCount == 0 else totalRetweetCount / totalTweetsCount
        traditionalFinalFeatures['url_ratio'] = 0 if totalTweetsCount == 0 else totalUrlCount / totalTweetsCount
        traditionalFinalFeatures['mentions_ratio'] = 0 if totalTweetsCount == 0 else totalMentionCount / totalTweetsCount
        traditionalFinalFeatures['retweet_per_tweet'] = 0 if totalTweetsCount == 0 else totalRetweetToTweetsCount / totalTweetsCount
        traditionalFinalFeatures['favourites_per_tweet'] = 0 if totalTweetsCount == 0 else totalLikesToTweetsCount / totalTweetsCount

        return traditionalFinalFeatures

    d = {'id': [user.id]} 
    dataframe = pd.DataFrame(data=d)
    dataframe = dataframe.apply(extractTweetFeatures, axis=1)
    dataframe = dataframe.drop(["id"], axis = 1)
    dataframe['geo_enabled'] = user.geo_enabled
    dataframe['statuses_count'] = user.statuses_count
    dataframe['followers_count'] = user.followers_count
    dataframe['friends_count'] = user.friends_count
    dataframe['favourites_count'] = user.favourites_count
    dataframe['user_name_length'] = len(user.name)
    dataframe['screen_name_length'] = len(user.screen_name)
    dataframe['description_length'] = len(user.description)
    dataframe['follower_to_friends_ratio'] = followersToFriendsRatio(user.followers_count, user.friends_count)
    dataframe['reputation_score'] = reputationScore(user.followers_count, user.friends_count)
    dataframe['lev_distance'] = lev(user.screen_name,user.name)
    dataframe['age'] = 2022 - user.created_at.year
    dataframe['tweet_to_age_ratio'] = 1 if user.created_at.year == 2022 else user.statuses_count / (2022 - user.created_at.year)
    

    print(dataframe)

    feature_vector = [dataframe['TTIM'], dataframe['TTISD'], dataframe['TTM'], dataframe['TTSD'], dataframe['avg_tweet_size'],
                    dataframe['avg_hashtag_count'], dataframe['retweeted_to_total_tweets'], dataframe['url_ratio'], dataframe['mentions_ratio'],
                    dataframe['retweet_per_tweet'], dataframe['favourites_per_tweet'], dataframe['geo_enabled'], dataframe['statuses_count'], 
                    dataframe['followers_count'], dataframe['friends_count'], dataframe['favourites_count'], dataframe['user_name_length'], 
                    dataframe['screen_name_length'], dataframe['description_length'], dataframe['follower_to_friends_ratio'], 
                    dataframe['reputation_score'], dataframe['lev_distance'], dataframe['age'], dataframe['tweet_to_age_ratio']]

    print(feature_vector)

    prediction = model.predict(dataframe) #PREDICT HERE
    #Round the output to 2 decimal place
    #If the output is negative, the values entered are unreasonable to the context of the application
    #If the output is greater than 0, return prediction
    print("Prediction: ")
    print(prediction)
    if prediction == 0:
        return render_template('index.html', prediction_text = "User " + Account_name + " is predicted as a human.")
    elif prediction == 1:
        return render_template('index.html', prediction_text = "User " + Account_name + " is predicted as a bot.")   

#Run app
if __name__ == "__main__":
    app.run(debug=True)