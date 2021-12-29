 | Idx | Table      | Column | Primary Key | Foreign Key | 
 | ----------- | ----------- | ----------- | ----------- | ----------- | 
  | 0 |  | * |   |   | 
 | 1 | **Addresses** | Address_ID | + | --> 18 | 
 | 2 |   | address_details |   |   | 
 | 3 | **Locations** | Location_ID | + | --> 22 | 
 | 4 |   | Other_Details |   |   | 
 | 5 | **Products** | Product_ID | + | --> 25 | 
 | 6 |   | Product_Type_Code |   |   | 
 | 7 |   | Product_Name |   |   | 
 | 8 |   | Product_Price |   |   | 
 | 9 | **Parties** | Party_ID | + | --> 26 | 
 | 10 |   | Party_Details |   |   | 
 | 11 | **Assets** | Asset_ID | + |   | 
 | 12 |   | Other_Details |   |   | 
 | 13 | **Channels** | Channel_ID | + |   | 
 | 14 |   | Other_Details |   |   | 
 | 15 | **Finances** | Finance_ID | + | --> 21 | 
 | 16 |   | Other_Details |   |   | 
 | 17 | **Events** | Event_ID | + | --> 27 | 
 | 18 |   | Address_ID |   |   | 
 | 19 |   | Channel_ID |   | --> 13 | 
 | 20 |   | Event_Type_Code |   |   | 
 | 21 |   | Finance_ID |   |   | 
 | 22 |   | Location_ID |   |   | 
 | 23 | **Products_in_Events** | Product_in_Event_ID | + |   | 
 | 24 |   | Event_ID |   |   | 
 | 25 |   | Product_ID |   |   | 
 | 26 | **Parties_in_Events** | Party_ID | + |   | 
 | 27 |   | Event_ID |   |   | 
 | 28 |   | Role_Code |   |   | 
 | 29 | **Agreements** | Document_ID | + |   | 
 | 30 |   | Event_ID |   |   | 
 | 31 | **Assets_in_Events** | Asset_ID | + | --> 11 | 
 | 32 |   | Event_ID |   |   | 
 
  | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |
  | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | 
 | SPIDER_train_4584 | List the name of products in ascending order of price. | SELECT Product_Name FROM Products ORDER BY Product_Price ASC | 1. SELECT[tbl:​Products] <br>2. PROJECT[col:​Products:​Product_Name, #1] <br>3. PROJECT[col:​Products:​Product_Price, #1] <br>4. SORT[#2, #3, sortdir:​ascending] <br> | 1. SELECT[tbl:​Products] <br>2. PROJECT[col:​Products:​Product_Name, #1] <br>3. PROJECT[col:​Products:​Product_Price, #1] <br>4. SORT[#2, #3, sortdir:​ascending] <br> | + | easy | 
  | SPIDER_train_4585 | What are the names and type codes of products? | SELECT Product_Name ,  Product_Type_Code FROM Products | 1. SELECT[tbl:​Products] <br>2. PROJECT[col:​Products:​Product_Name, #1] <br>3. PROJECT[col:​Products:​Product_Type_Code, #1] <br>4. UNION[#2, #3] <br> | 1. SELECT[tbl:​Products] <br>2. PROJECT[col:​Products:​Product_Name, #1] <br>3. PROJECT[col:​Products:​Product_Type_Code, #1] <br>4. UNION[#2, #3] <br> | + | medium | 
  | SPIDER_train_4586 | Show the prices of the products named "Dining" or "Trading Policy". | SELECT Product_Price FROM Products WHERE Product_Name  =  "Dining" OR Product_Name  =  "Trading Policy" | 1. SELECT[tbl:​Products] <br>2. FILTER[#1, comparative:​=:​Dining:​col:​Products:​Product_Name] <br>3. FILTER[#1, comparative:​=:​Trading Policy:​col:​Products:​Product_Name] <br>4. UNION[#2, #3] <br>5. PROJECT[col:​Products:​Product_Price, #4] <br> | 1. SELECT[tbl:​Products] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Dining:​col:​Products:​Product_Name] <br>3. COMPARATIVE[#1, #1, comparative:​=:​Trading Policy:​col:​Products:​Product_Name] <br>4. UNION[#2, #3] <br>5. PROJECT[col:​Products:​Product_Price, #4] <br> | + | medium | 
  | SPIDER_train_4587 | What is the average price for products? | SELECT avg(Product_Price) FROM Products | 1. SELECT[tbl:​Products] <br>2. PROJECT[col:​Products:​Product_Price, #1] <br>3. AGGREGATE[avg, #2] <br> | 1. SELECT[tbl:​Products] <br>2. PROJECT[col:​Products:​Product_Price, #1] <br>3. AGGREGATE[avg, #2] <br> | + | easy | 
  | SPIDER_train_4588 | What is the name of the product with the highest price? | SELECT Product_Name FROM Products ORDER BY Product_Price DESC LIMIT 1 | 1. SELECT[tbl:​Products] <br>2. PROJECT[col:​Products:​Product_Price, #1] <br>3. COMPARATIVE[#1, #2, comparative:​max:​None] <br>4. PROJECT[col:​Products:​Product_Name, #3] <br> | 1. SELECT[tbl:​Products] <br>2. PROJECT[col:​Products:​Product_Price, #1] <br>3. SUPERLATIVE[comparative:​max:​None, #1, #2] <br>4. PROJECT[col:​Products:​Product_Name, #3] <br> | + | medium | 
  | SPIDER_train_4589 | Show different type codes of products and the number of products with each type code. | SELECT Product_Type_Code ,  COUNT(*) FROM Products GROUP BY Product_Type_Code | 1. SELECT[col:​Products:​Product_Type_Code] <br>2.*(distinct)* PROJECT[different #REF, #1] <br>3. PROJECT[tbl:​Products, #2] <br>4. GROUP[count, #3, #1] <br>5. UNION[#2, #4] <br> | 1. SELECT[col:​Products:​Product_Type_Code] <br>2.*(distinct)* PROJECT[None, #1] <br>3. PROJECT[tbl:​Products, #2] <br>4. GROUP[count, #3, #1] <br>5. UNION[#2, #4] <br> | + | medium | 
  | SPIDER_train_4590 | Show the most common type code across products. | SELECT Product_Type_Code FROM Products GROUP BY Product_Type_Code ORDER BY COUNT(*) DESC LIMIT 1 | 1. SELECT[col:​Products:​Product_Type_Code] <br>2. PROJECT[tbl:​Products, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | 1. SELECT[col:​Products:​Product_Type_Code] <br>2. PROJECT[tbl:​Products, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | + | hard | 
  | SPIDER_train_4591 | Show the product type codes that have at least two products. | SELECT Product_Type_Code FROM Products GROUP BY Product_Type_Code HAVING COUNT(*)  >=  2 | 1. SELECT[col:​Products:​Product_Type_Code] <br>2. PROJECT[tbl:​Products, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​2] <br> | 1. SELECT[col:​Products:​Product_Type_Code] <br>2. PROJECT[tbl:​Products, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​2] <br> | + | easy | 
  | SPIDER_train_4592 | Show the product type codes that have both products with price higher than 4500 and products with price lower than 3000. | SELECT Product_Type_Code FROM Products WHERE Product_Price  >  4500 INTERSECT SELECT Product_Type_Code FROM Products WHERE Product_Price  <  3000 | 1. SELECT[tbl:​Products] <br>2. FILTER[#1, comparative:​<:​3000:​col:​Products:​Product_Price] <br>3. FILTER[#1, comparative:​>:​4500:​col:​Products:​Product_Price] <br>4. PROJECT[col:​Products:​Product_Type_Code, #1] <br>5. INTERSECTION[#4, #2, #3] <br> | 1. SELECT[tbl:​Products] <br>2. COMPARATIVE[#1, #1, comparative:​<:​3000:​col:​Products:​Product_Price] <br>3. COMPARATIVE[#1, #1, comparative:​>:​4500:​col:​Products:​Product_Price] <br>4. PROJECT[col:​Products:​Product_Type_Code, #1] <br>5. INTERSECTION[#4, #2, #3] <br> | + | hard | 
  | SPIDER_train_4593 | Show the names of products and the number of events they are in. | SELECT T1.Product_Name ,  COUNT(*) FROM Products AS T1 JOIN Products_in_Events AS T2 ON T1.Product_ID  =  T2.Product_ID GROUP BY T1.Product_Name | 1. SELECT[col:​Products:​Product_Name] <br>2. PROJECT[col:​Products:​Product_Name, #1] <br>3. PROJECT[tbl:​Products_in_Events, #1] <br>4. GROUP[count, #3, #1] <br>5. UNION[#2, #4] <br> | 1. SELECT[col:​Products:​Product_Name] <br>2. PROJECT[col:​Products:​Product_Name, #1] <br>3. PROJECT[tbl:​Products_in_Events, #1] <br>4. GROUP[count, #3, #1] <br>5. UNION[#2, #4] <br> | + | medium | 
  | SPIDER_train_4595 | Show the names of products that are in at least two events. | SELECT T1.Product_Name FROM Products AS T1 JOIN Products_in_Events AS T2 ON T1.Product_ID  =  T2.Product_ID GROUP BY T1.Product_Name HAVING COUNT(*)  >=  2 | 1. SELECT[tbl:​Products] <br>2. SELECT[tbl:​Products_in_Events] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​2] <br>5. PROJECT[col:​Products:​Product_Name, #4] <br> | 1. SELECT[tbl:​Products] <br>2. SELECT[tbl:​Products_in_Events] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​2] <br>5. PROJECT[col:​Products:​Product_Name, #4] <br> | + | medium | 
  | SPIDER_train_4596 | Show the names of products that are in at least two events in ascending alphabetical order of product name. | SELECT T1.Product_Name FROM Products AS T1 JOIN Products_in_Events AS T2 ON T1.Product_ID  =  T2.Product_ID GROUP BY T1.Product_Name HAVING COUNT(*)  >=  2 ORDER BY T1.Product_Name | 1. SELECT[tbl:​Products] <br>2. PROJECT[tbl:​Products_in_Events, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​2] <br>5. PROJECT[col:​Products:​Product_Name, #4] <br>6. SORT[#5, #5, sortdir:​ascending] <br> | 1. SELECT[tbl:​Products] <br>2. PROJECT[tbl:​Products_in_Events, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​2] <br>5. PROJECT[col:​Products:​Product_Name, #4] <br>6. SORT[#5, #5, sortdir:​ascending] <br> | + | hard | 
  | SPIDER_train_4597 | List the names of products that are not in any event. | SELECT Product_Name FROM Products WHERE Product_ID NOT IN (SELECT Product_ID FROM Products_in_Events) | 1. SELECT[tbl:​Products_in_Events] <br>2. PROJECT[tbl:​Products, #1] <br>3. SELECT[tbl:​Products] <br>4. DISCARD[#3, #2] <br>5. PROJECT[col:​Products:​Product_Name, #4] <br> | 1. SELECT[tbl:​Products_in_Events] <br>2. PROJECT[tbl:​Products, #1] <br>3. SELECT[tbl:​Products] <br>4. DISCARD[#3, #2] <br>5. PROJECT[col:​Products:​Product_Name, #4] <br> | + | hard | 
 ***
 Exec acc: **1.0000**
