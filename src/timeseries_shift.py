# encoding=utf-8

# PySpark extension for IBM SPSS Modeler 18.0 and higher

# shift values down (or up) a list of columns  and insert a NULL value into the first (or last) column

import sys

ascontext = None
if len(sys.argv) > 1 and sys.argv[1] == "-test":
    import pyspark
    sc = pyspark.SparkContext()
    sql = pyspark.sql.SQLContext(sc)
    df = sql.read.format("com.databricks.spark.csv").option("header", "true").load("example/PRSA_data_24_hr_windows_by_year.csv")
    window_fields = ["pm2point5"]+["pm2point5-"+str(offset) for offset in range(1,24)]
    shiftDown=True
else:
    import spss.pyspark.runtime
    ascontext = spss.pyspark.runtime.getContext()
    sc = ascontext.getSparkContext()
    sql = ascontext.getSparkSQLContext()
    df = ascontext.getSparkInputData()
    schema = ascontext.getSparkInputSchema()
    window_fields = filter(lambda x: len(x) > 0, map(lambda x: x.strip(), "%%window_fields%%".split(",")))
    shiftDown=True
    if '%%shift_values%%' == 'up':
        shiftDown=False

if ascontext and ascontext.isComputeDataModelOnly():
    ascontext.setSparkOutputSchema(schema)
    sys.exit(0)

outdf = df

from pyspark.sql.functions import lit, when, col

def make_null(x):
    return when(col(x) == None, col(x)).otherwise(lit(None))

if shiftDown:
    for i in range(len(window_fields)-1,0,-1):
        outdf = outdf.withColumn(window_fields[i],outdf[window_fields[i-1]])
    first_col = window_fields[0]
    outdf = outdf.withColumn(first_col,make_null(first_col))
else:
    for i in range(0,len(window_fields)-1):
        outdf = outdf.withColumn(window_fields[i],outdf[window_fields[i+1]])
    last_col = window_fields[len(window_fields)-1]
    outdf = outdf.withColumn(last_col, make_null(last_col))

if ascontext:
    ascontext.setSparkOutputData(outdf)
else:
    print(outdf.dtypes)
    outdf.show()
