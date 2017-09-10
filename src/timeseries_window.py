# encoding=utf-8
import sys
import pyspark

# PySpark extension for IBM SPSS Modeler 18.0 and higher

# convert data from a time series format to a rolling window format (with a configurable window size)
# optional - specify one or more grouping fields.
#
# Input format
#
# [grp0,...,grpM],order0,...,orderN, series1,...,seriesX
#
# Output format [window size W]
#
# INDEX,[grp0,...,grpM],order0,...,orderN,series1,...,seriesX,series1-1,...,series1-W,...,seriesX-1,...,seriesX-W
#

from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField

ascontext = None
if len(sys.argv) > 1 and sys.argv[1] == "-test":
    sc = pyspark.SparkContext()
    sql = pyspark.sql.SQLContext(sc)
    hsql = pyspark.sql.HiveContext(sc)
    df = sql.read.format("com.databricks.spark.csv").option("header", "true").load("example/PRSA_data_2010.1.1-2014.12.31.csv")
    df = df.withColumn("year", df["year"].cast(IntegerType()))
    df = df.withColumn("month", df["month"].cast(IntegerType()))
    df = df.withColumn("day", df["day"].cast(IntegerType()))
    df = df.withColumn("hour", df["hour"].cast(IntegerType()))
    df = df.withColumn("pm2point5", df["pm2point5"].cast(DoubleType()))
    df = df.withColumn("TEMP", df["TEMP"].cast(DoubleType()))
    schema = df.schema
    group_fields = []
    order_fields = ["year","month","day","hour"]
    data_fields = ["pm2point5","TEMP"]
    window_size = 3
    index_field_name = "INDEX"
else:
    import spss.pyspark.runtime
    ascontext = spss.pyspark.runtime.getContext()
    sc = ascontext.getSparkContext()
    sql = ascontext.getSparkSQLContext()

    schema = ascontext.getSparkInputSchema()
    df = ascontext.getSparkInputData()
    group_fields = filter(lambda x:len(x)>0,map(lambda x: x.strip(),"%%group_fields%%".split(",")))
    order_fields = filter(lambda x:len(x)>0,map(lambda x: x.strip(),"%%order_fields%%".split(",")))
    data_fields = filter(lambda x:len(x)>0,map(lambda x: x.strip(),"%%data_fields%%".split(",")))
    window_size = int('%%window_size%%')
    index_field_name = '%%index_field_name%%'
    if index_field_name == "":
        index_field_name = "INDEX"

def getField(schema,name):
    for field in schema.fields:
        if field.name == name:
            return field
    raise Exception("Could not find field %s"%(name))

output_schema = StructType([StructField(index_field_name, IntegerType(), nullable=True)]+
    [getField(schema,group_field) for group_field in group_fields]+
    [getField(schema,order_field) for order_field in order_fields]+
    [getField(schema,data_field) for data_field in data_fields]+
    [StructField(data_field+"-"+str(offset),getField(schema,data_field).dataType,nullable=True) for data_field in data_fields for offset in range(1,window_size)])

if ascontext and ascontext.isComputeDataModelOnly():
    ascontext.setSparkOutputSchema(output_schema)
    sys.exit(0)

outdf = df
if sc.version.split(".")[0] == "2":
    pdf = df.toPandas()

    pdf2 = pdf[order_fields+group_fields+data_fields].sort_values(order_fields,ascending=False).reset_index(drop=True)
    if len(group_fields):
        pdfg = pdf2.groupby(group_fields)
        pdf2[index_field_name] = pdfg.cumcount()+1
    else:
        pdfg = pdf2
        pdf2[index_field_name] = pdf2.index+1

    lead_fields = []
    for data_field in data_fields:
        for offset in range(1,window_size):
            lead_field_name = data_field+"-"+str(offset)
            pdf2[lead_field_name] = pdfg[data_field].shift(-offset)
            lead_fields.append(lead_field_name)

    outdf = sql.createDataFrame(pdf2[[index_field_name]+group_fields+order_fields+data_fields+lead_fields],output_schema)

else:
    from pyspark.sql.functions import lead
    from pyspark.sql.window import Window

    w = Window().orderBy([df[order_field].desc() for order_field in order_fields])\
        .partitionBy([group_field for group_field in group_fields])

    cols = [pyspark.sql.functions.row_number().over(w).alias(index_field_name)]
    cols += [df[group_field] for group_field in group_fields]
    cols += [df[order_field] for order_field in order_fields]
    cols += [df[data_field] for data_field in data_fields]
    cols += [lead(data_field,count=offset).over(w).alias(data_field+"-"+str(offset)) for data_field in data_fields for offset in range(1,window_size)]

    outdf = df.select(*cols)

outdf = outdf.na.drop()

if ascontext:
    ascontext.setSparkOutputData(outdf)
else:
    outdf.show()
