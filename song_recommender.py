import pyspark.mllib
from pyspark.sql import *
from pyspark import *
from pyspark.rdd import *
from pyspark.ml import *
from pyspark.ml.feature import *
from pyspark.ml.classification import *
from pyspark.sql.types import *
from pyspark.mllib.recommendation import *
import random


sc = SparkContext("local","music")
spark = SparkSession(sc)

sampleUsersPath = "sampleUsers.txt"
sampleTracksPath = "sampleTracks.txt"


# load data file
def load(path):
    schema = StructType([
                        StructField("user", StringType()), 
                        StructField("song", StringType()),
                        StructField("count", IntegerType())
    ])
    df = spark.read.text(path)
    rddFromDF = df.rdd.map(lambda row: row.value.split('\t'))
    rddFromDF = rddFromDF.map(lambda row: [row[0], row[1], int(row[2]) if int(row[2]) <= 20 else 20])
    dff = spark.createDataFrame(rddFromDF, schema)
    return dff
    raise NotImplementedError()
    
loaded = load(sampleUsersPath).persist()


# Schema type test
correctCols = StructType([\
StructField("user",StringType(),True),\
StructField("song",StringType(),True),\
StructField("count",IntegerType(),True)])

fakeData = [("abc123", "123abc", 2)]
fakeDf = spark.createDataFrame(fakeData, correctCols)
assert loaded.dtypes == fakeDf.dtypes, "the schema was expected to be %s but it was %s" % (fakeDf.dtypes, loaded.dtypes)
assert loaded.filter('count>20').count() == 0, "counts higher than 20 was expected to be 0 but it was %s" % loaded.filter('count>20').count()
test1 = str(loaded.sample(False, 0.01, seed=123).limit(1).first())
correct1 = "Row(user='a58de017cbeda1763ea002fe027ed41b4ed53109', song='SOJTLHS12A8C13F633', count=3)"
assert test1 == correct1, "the row was expected to be %s but it was %s" % (correct1, test1)


# Convert dataframe and train stringIndexer model
def convert(df):
    inputc = ["user", "song"]
    outputc = ["user_indexed", "song_indexed"]
    stringIndexer = StringIndexer(inputCols=inputc, outputCols=outputc)
    model = stringIndexer.fit(df)
    result = model.transform(df)
    return result
    raise NotImplementedError()
    
converted = convert(loaded).persist()


# Convert test
correctCols = StructType([\
StructField("user",StringType(),True),\
StructField("song",StringType(),True),\
StructField("count",IntegerType(),True),\
StructField("user_indexed",DoubleType(),True),\
StructField("song_indexed",DoubleType(),True)])

fakeData = [("abc123", "123abc", 2, 1.0, 2.0)]
fakeDf = spark.createDataFrame(fakeData, correctCols)
assert converted.dtypes == fakeDf.dtypes, "the schema was expected to be %s but it was %s" % (fakeDf.dtypes, converted.dtypes)

test2 = str(converted.sample(False, 0.1, seed=1234).limit(1).first())
correct2 = "Row(user='5a905f000fc1ff3df7ca807d57edb608863db05d', song='SOCHPFL12AF72A3F64', count=2, user_indexed=5.0, song_indexed=767.0)"
assert test2 == correct2, "the row was expected to be %s but it was %s" % (correct2, test2)


# Assign rating
def toRating(df):
    rdd = df.rdd.map(lambda row: Rating(user=row[3],product=row[4],rating=row[2]))
    return rdd
    raise NotImplementedError()
    
rated = toRating(converted).persist()


# Rating tests
rows = [Rating(user=162, product=577, rating=2.0),
 Rating(user=162, product=1053, rating=1.0),
 Rating(user=162, product=1646, rating=1.0),
 Rating(user=162, product=1945, rating=1.0),
 Rating(user=162, product=2306, rating=1.0)]
assert rated.take(5) == rows, "the first 5 rows were expected to be %s but they were %s" % (rows, rated.take(5))

random.seed(54321)
r = random.randint(100, 2000)

test3 = str(toRating(converted).collect()[r])
correct3 = "Rating(user=599, product=1321, rating=1.0)"
assert test3 == correct3, "the row was expected to be %s but it was %s" % (correct3, test3)

# Train ALS model
def trainALS(data, seed):
    rank = 10
    model = ALS.train(data, rank, seed=seed)
    return model
    raise NotImplementedError()
    
random.seed(123)
rSeed = random.randint(0, 10000)
model = trainALS(rated, rSeed)

# Recommend Songs
def recommendSongs(model, user):
    top5 = model.recommendProducts(user,5)
    return top5
    raise NotImplementedError()
    
recommends = recommendSongs(model, 162)

# Recommend songs test
assert type(recommends[0]) == pyspark.mllib.recommendation.Rating, "the type was expected to be pyspark.mllib.recommendation.Rating  but it was %s" % type(recommends[0]) 
assert recommends[0].user == 162, "the user was expected to be 162 but it was %s" % recommends[0].user
assert len(recommends) == 5, "the amount of recommendations was expected to be 5 but it was %s" % len(recommends)


# Get songs name
def getSongNames(converted, ar, path):
    schema = StructType([
                        StructField("track_id", StringType()), 
                        StructField("song_id", StringType()),
                        StructField("artist", StringType()),
                        StructField("title", StringType())
    ])
    
    tracks = spark.read.text(path)
    tracksRDD = tracks.rdd.map(lambda row: row.value.split('<SEP>'))
    arr = []
    for each in ar:
        arr.append(each[1])
    songs = converted.rdd.filter(lambda song: song[4] in arr).toDF()
    tracksDF = spark.createDataFrame(tracksRDD, schema)
    joinedSong = tracksDF.join(songs, tracksDF["song_id"] == songs["song"], how="inner")
    joinedSong = joinedSong.select("title", "artist").distinct()
    rec_song = joinedSong.rdd.map(lambda row: [row[0], row[1]]).collect()
    return rec_song
    raise NotImplementedError()
    
songNames = getSongNames(converted, recommends, sampleTracksPath)

# Get songs name test
'''getSongNames test'''
assert len(songNames) == 5, "the amount of song names was expected to be 5 but it was %s" % len(songNames)
assert type(songNames[0]) == list, "the type of a songNames element was expected to be list but it was %s" % type(songNames[0])
test5 = songNames[3]
correct5 = ['Awakenings', 'Symphony X']
assert test5 == correct5, "the row was expected to be %s but it was %s" % (correct5, test5)


# Get 5 recommended songs
def recommend(path, userId, tracksPath, seed):
    userDF = load(path)
    convDF = convert(userDF)
    toRateRDD = toRating(convDF)
    model = trainALS(toRateRDD, seed)
    user = convDF.where(convDF["user"]==userId)
    user = user.first().user_indexed
    recos = recommendSongs(model, int(user))
    rec_songs = getSongNames(convDF, recos, tracksPath)
    return rec_songs
    raise NotImplementedError()
    

recom = recommend(sampleUsersPath, "b80344d063b5ccb3212f76538f3d9e43d87dca9e" ,sampleTracksPath, rSeed)
print(recom)

# Get 5 songs test
assert len(recom) == 5, "the amount of recommendations was expected to be 5 but it was %s" % len(recom)
assert type(recom[0]) == list, "the type of a 'recommend' element was expected to be list but it was %s" % type(recom[0])
#test if the same user and seed returns the same as songNames
assert recom == songNames, "the song names were expected to be %s but they were %s" % (songNames, recom)
    