some scripts to convert a ROOT geometry description into a (MySQL) database

usage:

a) convert a ROOT file ( example cms2015.root ) into sql commands using

root -b -q -l ConvertROOTtoSQL.C("cms2015")


resulting in a file output.sql


b) create a database schema using file create_Table.sql 

(something like)
mysql < create_Table.sql


c) insert data into the database

(something like)
mysql < output.sql


