from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from numpy import array
from pyspark import SparkConf, SparkContext

import collections

# Create a SparkSession (Note, the config section is only for Windows!)
#spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("TotalPercentageCoverageSQL").getOrCreate()
spark = SparkSession.builder.appName("RadioCVDecision").getOrCreate()

def LTEmapper(line):
    fields = line.split(';')
    return Row(Cor1=int(fields[0]), dbm1=int(fields[1]))
    
def UserSpeedmapper(line):
    fields = line.split(';')
    return Row(Cor2=int(fields[0]), dbm2=int(fields[1]), speed2=int(fields[1]))

def LTETechThresmapper(line):
    fields = line.split(';')
    return Row(ltelt=int(fields[0]), lteht=int(fields[1]), ltels=int(fields[2]), ltehs=int(fields[3]))
    
    #Expected_udf = udf(lambda dbm1,speed2: "Y" if dbm1 <= -111 and dbm1 >= -140 and (speed2*60)/100 <= 44700 else ("Y" if dbm1 <= -91 and dbm1 >= -110 and (speed2*60)/100 <= 44700 else ("Y" if dbm1 <= -40 and dbm1 >= -90 and (speed2*60)/100 <= 60000 else "N")), StringType())

def ExpectedResult(DBMInRange, SpeedPerct):
    if (DBMInRange == 1 and SpeedPerct == 1):
        return 1
    elif (DBMInRange == 1 and SpeedPerct == 0):
        return 0
    elif (DBMInRange == 0 and SpeedPerct == 1):
        return 1
    elif (DBMInRange == 0 and SpeedPerct == 0):
        return 0 
    else: 
        return 0          
        
def binary(YN):
    if (YN == 'Y'):
        return 1
    else:
        return 0

def createLabeledPoints(fields):
    XYCor = int(fields[0])
    DBMInRange = int(fields[1])
    SpeedPerct = int(fields[3])
    ExpectedVal = ExpectedResult(int(fields[1]), int(fields[3]))
    #print ("$$$$$$$$$$$$")
    #print XYCor
    #print fields[1]
    #print DBMInRange
    #print fields[3]
    #print SpeedPerct
    #print ExpectedVal
    #print ("$$$$$$$$$$$$")
    return LabeledPoint(ExpectedVal, array([int(XYCor), DBMInRange, SpeedPerct]))
    
    
def createActualPoints(fields):
    XYCor = int(fields[0])
    DBMInRange = int(fields[1])
    SpeedPerct = int(fields[3])
    ExpectedVal = ExpectedResult(int(fields[1]), int(fields[3]))
    #print ("$$$$$$$$$$$$")
    #print XYCor
    #print fields[1]
    #print DBMInRange
    #print fields[3]
    #print SpeedPerct
    #print ExpectedVal
    #print ("$$$$$$$$$$$$")
    return (array([int(XYCor), DBMInRange, SpeedPerct, ExpectedVal]))
   
   
lines1 = spark.sparkContext.textFile("LTE_L800_SS.csv")
lines2 = spark.sparkContext.textFile("UserSpeedLTE800.csv")
lines3 = spark.sparkContext.textFile("LTETechThresholds.csv")

lines4 = spark.sparkContext.textFile("LTE800CV1.csv")
lines5 = spark.sparkContext.textFile("UserSpeed1.csv")


Vals1 = lines1.map(LTEmapper)
Vals2 = lines2.map(UserSpeedmapper)
Vals3 = lines3.map(LTEmapper)

Vals4 = lines4.map(LTEmapper)
Vals5 = lines5.map(UserSpeedmapper)

# Infer the schema, and register the DataFrame as a table.
schemaVal1 = spark.createDataFrame(Vals1)
schemaVal1.createOrReplaceTempView("Vals1")

schemaVal2 = spark.createDataFrame(Vals2)
schemaVal2.createOrReplaceTempView("Vals2")

schemaVal3 = spark.createDataFrame(Vals3)#.cache()
schemaVal3.createOrReplaceTempView("Vals3")


schemaVal4 = spark.createDataFrame(Vals4)#.cache()
schemaVal4.createOrReplaceTempView("Vals4")

schemaVal5 = spark.createDataFrame(Vals5)#.cache()
schemaVal5.createOrReplaceTempView("Vals5")


# SQL can be run over DataFrames that have been registered as a table.

#spark.sql("SELECT * FROM Vals WHERE dbm >= -90.00").count()
#DBMVal1 = spark.sql("SELECT count (*) FROM Vals1 WHERE dbm >= -90.00")
#results1 = DBMVal1.collect()

#DBMVal2 = spark.sql("SELECT count (*) FROM Vals2 WHERE dbm >= -68.00")
#results2 = DBMVal2.collect()

#DBMVal3 = spark.sql("SELECT count (*) FROM Vals3 WHERE dbm >= -102.00")
#results3 = DBMVal3.collect()

DBMVal1 = spark.sql("SELECT * FROM Vals1 v1 JOIN Vals2 v2 ON v1.Cor1 = v2.Cor2")

#ON v1.Cor = v2.Cor AND v1.Cor = v3.Cor AND v1.Cor = v4.Cor ")
results1 = DBMVal1.collect()


DBMVal2 = spark.sql("SELECT * FROM Vals4 v4 JOIN Vals5 v5 ON v4.Cor1 = v5.Cor2")

#ON v1.Cor = v2.Cor AND v1.Cor = v3.Cor AND v1.Cor = v4.Cor ")
results2 = DBMVal2.collect()

#for result in results1:
 #   print result[0]
  #  print result[1]
   # print result[2]
    #print result[3]
    #print result[4]
    #print 9result[5]
    #print result[6]
    #print ("*******************")

DBMInRange_udf = udf(lambda dbm1,dbm2: "1" if dbm1 <= dbm2 else "0", StringType())
df = spark.createDataFrame(results1)
dfoutput = df.select((df.Cor1).alias('dfCor'), DBMInRange_udf(df.dbm1,df.dbm2).alias('DBM_In_Range'))

res1 = spark.createDataFrame(results1)

dfactual = spark.createDataFrame(results2)
dfoutputactual = dfactual.select((dfactual.Cor1).alias('dfactualCor'), DBMInRange_udf(dfactual.dbm1,dfactual.dbm2).alias('DBM_In_Range'))

resactual = spark.createDataFrame(results2)


#-43;-90;3800;60000
#-91;-110;3250;49000
#-111;-140;2900;44700

#PercentSpeedRange_udf =  udf(lambda dbm1,speed1,speed2: "Y" if speed1 <= -111 and dbm1 >= -140 and speed2 >= 2900 and speed2 <= 44700 else ("Y" if dbm1 <= -91 and dbm1 >= -110 and speed2 >= 3250 and speed2 <= 44700 else ("Y" if dbm1 <= -40 and dbm1 >= -90 and speed2 >= 3800 and speed2 <= 60000 else "N")), StringType())

#SpeedInRange_udf = udf(lambda dbm1,speed2: "Y" if dbm1 <= -111 and dbm1 >= -140 and speed2 >= 2900 and speed2 <= 44700 else ("Y" if dbm1 <= -91 and dbm1 >= -110 and speed2 >= 3250 and speed2 <= 44700 else ("Y" if dbm1 <= -40 and dbm1 >= -90 and speed2 >= 3800 and speed2 <= 60000 else "N")), StringType())
SpeedPerct_udf = udf(lambda dbm1,speed2: "1" if dbm1 <= -111 and dbm1 >= -140 and (speed2*60)/100 <= 44700 else ("0" if dbm1 <= -91 and dbm1 >= -110 and (speed2*60)/100 <= 44700 else ("1" if dbm1 <= -40 and dbm1 >= -90 and (speed2*60)/100 <= 60000 else "0")), StringType())

#SpeedInRange_udf = udf(lambda dbm1,speed2: "Y" if dbm1 <= -111 and dbm1 >= -140 and speed2 >= 2900 and speed2 <= 44700 else ("Y" if dbm1 <= -91 and dbm1 >= -110 and speed2 >= 3250 and speed2 <= 44700 else "N"), StringType())


sdf = spark.createDataFrame(results1)
sdfoutput = sdf.select(sdf.Cor1.alias("sdfCor"), SpeedPerct_udf(sdf.dbm1,sdf.speed2).alias('SpeedPerct'))#.show()

res2 = dfoutput.join(sdfoutput, dfoutput.dfCor == sdfoutput.sdfCor).collect()

resdf = spark.createDataFrame(res2)

resmap = resdf.rdd


sdfactual = spark.createDataFrame(results2)
sdfoutputactual  = sdfactual.select(sdfactual.Cor1.alias("sdfactualCor"), SpeedPerct_udf(sdfactual.dbm1,sdfactual.speed2).alias('SpeedPerct'))#.show()

res2actual = dfoutputactual.join(sdfoutputactual, dfoutputactual.dfactualCor == sdfoutputactual.sdfactualCor).collect()

resdfactual = spark.createDataFrame(res2actual)

resmapactual = resdfactual.rdd

#print ("*******************")
#for res in res2:
#    print res[0]
#    print res[1]
 #   print res[3]
    #print result[6]
    #print ("*******************")

trainingData = resmap.map(createLabeledPoints)

actualData = resmapactual.map(createActualPoints)
print (actualData)


print ("++++++++++++++++++++++++")

#conf = SparkConf()
#sc = SparkContext(conf = conf)
#testCandidates = [ array([507100149100,1, 0])]
testCandidates = actualData

#testData = spark.sparkContext.parallelize([testCandidates])
#for i in range(0,testCandidates.count()):
 #   testData = spark.sparkContext.parallelize((testCandidates[i]))
    #print ('I am here' + repr(testCandidates[i]))
    #print(model.toDebugString())
    
    #print ('after Model' + repr(testCandidates[i]))
    
    #rdd = df2.map(lambda data: Vectors.dense([float(c) for c in data]))
    


# Train our DecisionTree classifier using our data set
model = DecisionTree.trainClassifier(trainingData, numClasses=2,
                                     categoricalFeaturesInfo={1:2, 2:2},
                                     impurity='gini', maxDepth=5, maxBins=35)

# Now get predictions for our unknown candidates. (Note, you could separate
# the source data into a training set and a test set while tuning
# parameters and measure accuracy as you go!)
#predictions = model.predict(testData)


predictions = model.predict(actualData)


print ('Expected Coverage Results===========>')
results = predictions.collect()
for result in results:
    print ('Expected Coverage prediction:: ' + repr(result))
    # We can also print out the decision tree itself:
    print('Learned classification tree model:')
    print(model.toDebugString())


predictions.saveAsTextFile("Coverage_Predictions_Forest.txt")

#print ("################################")

#print (DBM_In_Range)

#print ("################################")

#JOIN Vals6 v6 JOIN Vals7 v7 JOIN Vals8 v8
#AND v1.Cor = v6.Cor AND v1.Cor = v7.Cor AND v1.Cor = v8.Cor



#DBMVal8 = spark.sql("SELECT * FROM Vals1 v1 JOIN Vals4 v4 ON v1.Cor = v4.Cor WHERE v1.dbm != V4.dbm")
#results8 = DBMVal8.collect()

#DBMVal9 = spark.sql("SELECT * FROM Vals4 v4 JOIN Vals4 v4 ON v1.Cor = v2.Cor AND v1.Cor = v3.Cor")
#results9 = DBMVal7.collect()



    


#Totpixel = (int(count)/100000)*100

#print("Total pixel covered :: " + str(Totpixel) + "%")

# The results of SQL queries are RDDs and support all the normal RDD operations.
#for val in DBMVal.collect():
 # print(val)

# We can also use functions instead of SQL queries:
#schemaPeople.groupBy("dbm").count().orderBy("dbm").show()

spark.stop()
