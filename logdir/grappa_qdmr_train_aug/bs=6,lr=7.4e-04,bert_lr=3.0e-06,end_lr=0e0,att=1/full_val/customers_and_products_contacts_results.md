 | Idx | Table      | Column | Primary Key | Foreign Key | 
 | ----------- | ----------- | ----------- | ----------- | ----------- | 
  | 0 |  | * |   |   | 
 | 1 | **Addresses** | address_id | + | --> 25 | 
 | 2 |   | line_1_number_building |   |   | 
 | 3 |   | city |   |   | 
 | 4 |   | zip_postcode |   |   | 
 | 5 |   | state_province_county |   |   | 
 | 6 |   | country |   |   | 
 | 7 | **Products** | product_id | + | --> 34 | 
 | 8 |   | product_type_code |   |   | 
 | 9 |   | product_name |   |   | 
 | 10 |   | product_price |   |   | 
 | 11 | **Customers** | customer_id | + | --> 24 | 
 | 12 |   | payment_method_code |   |   | 
 | 13 |   | customer_number |   |   | 
 | 14 |   | customer_name |   |   | 
 | 15 |   | customer_address |   |   | 
 | 16 |   | customer_phone |   |   | 
 | 17 |   | customer_email |   |   | 
 | 18 | **Contacts** | contact_id | + |   | 
 | 19 |   | customer_id |   | --> 11 | 
 | 20 |   | gender |   |   | 
 | 21 |   | first_name |   |   | 
 | 22 |   | last_name |   |   | 
 | 23 |   | contact_phone |   |   | 
 | 24 | **Customer_Address_History** | customer_id |   |   | 
 | 25 |   | address_id |   |   | 
 | 26 |   | date_from |   |   | 
 | 27 |   | date_to |   |   | 
 | 28 | **Customer_Orders** | order_id | + | --> 33 | 
 | 29 |   | customer_id |   |   | 
 | 30 |   | order_date |   |   | 
 | 31 |   | order_status_code |   |   | 
 | 32 | **Order_Items** | order_item_id |   |   | 
 | 33 |   | order_id |   |   | 
 | 34 |   | product_id |   |   | 
 | 35 |   | order_quantity |   |   | 
 
  | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |
  | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | 
 | SPIDER_train_5653 | How many addresses are there in country USA? | SELECT count(*) FROM addresses WHERE country  =  'USA' | 1. SELECT[tbl:​Addresses] <br>2. FILTER[#1, comparative:​=:​USA:​col:​Addresses:​country] <br>3. AGGREGATE[count, #2] <br> | 1. SELECT[tbl:​Addresses] <br>2. COMPARATIVE[#1, #1, comparative:​=:​USA:​col:​Addresses:​country] <br>3. AGGREGATE[count, #2] <br> | + | easy | 
  | SPIDER_train_5654 | Show all distinct cities in the address record. | SELECT DISTINCT city FROM addresses | 1. SELECT[tbl:​Addresses] <br>2. PROJECT[col:​Addresses:​city, #1] <br> | 1. SELECT[tbl:​Addresses] <br>2. PROJECT[col:​Addresses:​city, #1] <br> | + | easy | 
  | SPIDER_train_5655 | Show each state and the number of addresses in each state. | SELECT state_province_county ,  count(*) FROM addresses GROUP BY state_province_county | 1. SELECT[col:​Addresses:​state_province_county] <br>2. PROJECT[tbl:​Addresses, #1] <br>3. GROUP[count, #2, #1] <br>4. UNION[#1, #3] <br> | 1. SELECT[col:​Addresses:​state_province_county] <br>2. PROJECT[tbl:​Addresses, #1] <br>3. GROUP[count, #2, #1] <br>4. UNION[#1, #3] <br> | + | medium | 
  | SPIDER_train_5656 | Show names and phones of customers who do not have address information. | SELECT customer_name ,  customer_phone FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM customer_address_history) | 1. SELECT[tbl:​Customers] <br>2. FILTER[#1, tbl:​Customer_Address_History] <br>3. DISCARD[#1, #2] <br>4. PROJECT[col:​Customers:​customer_name, #3] <br>5. PROJECT[col:​Customers:​customer_phone, #3] <br>6. UNION[#4, #5] <br> | 1. SELECT[tbl:​Customers] <br>2. COMPARATIVE[#1, #1, tbl:​Customer_Address_History] <br>3. DISCARD[#1, #2] <br>4. PROJECT[col:​Customers:​customer_name, #3] <br>5. PROJECT[col:​Customers:​customer_phone, #3] <br>6. UNION[#4, #5] <br> | + | extra | 
  | SPIDER_train_5657 | Show the name of the customer who has the most orders. | SELECT T1.customer_name FROM customers AS T1 JOIN customer_orders AS T2 ON T1.customer_id  =  T2.customer_id GROUP BY T1.customer_id ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[col:​Customer_Orders:​customer_id] <br>2. PROJECT[tbl:​Customer_Orders, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​Customers:​customer_name, #4] <br> | 1. SELECT[col:​Customer_Orders:​customer_id] <br>2. PROJECT[tbl:​Customer_Orders, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​Customers:​customer_name, #4] <br> | + | extra | 
  | SPIDER_train_5658 | Show the product type codes which have at least two products. | SELECT product_type_code FROM products GROUP BY product_type_code HAVING count(*)  >=  2 | 1. SELECT[col:​Products:​product_type_code] <br>2. PROJECT[tbl:​Products, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​2] <br> | 1. SELECT[col:​Products:​product_type_code] <br>2. PROJECT[tbl:​Products, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​2] <br> | + | easy | 
  | SPIDER_train_5660 | Show the name, phone, and payment method code for all customers in descending order of customer number. | SELECT customer_name ,  customer_phone ,  payment_method_code FROM customers ORDER BY customer_number DESC | 1. SELECT[tbl:​Customers] <br>2. PROJECT[col:​Customers:​customer_name, #1] <br>3. PROJECT[col:​Customers:​customer_phone, #1] <br>4. PROJECT[col:​Customers:​payment_method_code, #1] <br>5. PROJECT[col:​Customers:​customer_number, #1] <br>6. UNION[#2, #3, #4] <br>7. SORT[#6, #5, sortdir:​descending] <br> | 1. SELECT[tbl:​Customers] <br>2. PROJECT[col:​Customers:​customer_name, #1] <br>3. PROJECT[col:​Customers:​customer_phone, #1] <br>4. PROJECT[col:​Customers:​payment_method_code, #1] <br>5. PROJECT[col:​Customers:​customer_number, #1] <br>6. UNION[#2, #3, #4] <br>7. SORT[#6, #5, sortdir:​descending] <br> | + | medium | 
  | SPIDER_train_5662 | Show the minimum, maximum, average price for all products. | SELECT min(product_price) ,  max(product_price) ,  avg(product_price) FROM products | 1. SELECT[tbl:​Products] <br>2. PROJECT[col:​Products:​product_price, #1] <br>3. AGGREGATE[min, #2] <br>4. AGGREGATE[max, #2] <br>5. AGGREGATE[avg, #2] <br>6. UNION[#3, #4, #5] <br> | 1. SELECT[tbl:​Products] <br>2. PROJECT[col:​Products:​product_price, #1] <br>3. AGGREGATE[min, #2] <br>4. AGGREGATE[max, #2] <br>5. AGGREGATE[avg, #2] <br>6. UNION[#3, #4, #5] <br> | + | medium | 
  | SPIDER_train_5663 | How many products have a price higher than the average? | SELECT count(*) FROM products WHERE product_price  >  (SELECT avg(product_price) FROM products) | 1. SELECT[tbl:​Products] <br>2. PROJECT[col:​Products:​product_price, #1] <br>3. AGGREGATE[avg, #2] <br>4. COMPARATIVE[#1, #2, comparative:​>:​#3] <br>5. AGGREGATE[count, #4] <br> | 1. SELECT[tbl:​Products] <br>2. PROJECT[col:​Products:​product_price, #1] <br>3. AGGREGATE[avg, #2] <br>4. COMPARATIVE[#1, #2, comparative:​>:​Sony:​col:​Products:​product_name] <br>5. AGGREGATE[count, #4] <br> | - | hard | 
  | SPIDER_train_5664 | Show the customer name, customer address city, date from, and date to for each customer address history. | SELECT T2.customer_name ,  T3.city ,  T1.date_from ,  T1.date_to FROM customer_address_history AS T1 JOIN customers AS T2 ON T1.customer_id  =  T2.customer_id JOIN addresses AS T3 ON T1.address_id  =  T3.address_id | 1. SELECT[tbl:​Customer_Address_History] <br>2. PROJECT[col:​Customers:​customer_name, #1] <br>3. PROJECT[col:​Addresses:​city, #1] <br>4. PROJECT[col:​Customer_Address_History:​date_from, #1] <br>5. PROJECT[col:​Customer_Address_History:​date_to, #1] <br>6. UNION[#2, #3, #4, #5] <br> | 1. SELECT[tbl:​Customer_Address_History] <br>2. PROJECT[col:​Customers:​customer_name, #1] <br>3. PROJECT[col:​Addresses:​city, #1] <br>4. PROJECT[col:​Customer_Address_History:​date_from, #1] <br>5. PROJECT[col:​Customer_Address_History:​date_to, #1] <br>6. UNION[#2, #3, #4, #5] <br> | + | medium | 
  | SPIDER_train_5665 | Show the names of customers who use Credit Card payment method and have more than 2 orders. | SELECT T1.customer_name FROM customers AS T1 JOIN customer_orders AS T2 ON T1.customer_id  =  T2.customer_id WHERE T1.payment_method_code  =  'Credit Card' GROUP BY T1.customer_id HAVING count(*)  >  2 | 1. SELECT[col:​Customer_Orders:​customer_id] <br>2. FILTER[#1, comparative:​=:​Credit Card:​col:​Customers:​payment_method_code] <br>3. PROJECT[tbl:​Customer_Orders, #2] <br>4. GROUP[count, #3, #2] <br>5. COMPARATIVE[#2, #4, comparative:​>:​2] <br>6. PROJECT[col:​Customers:​customer_name, #5] <br> | 1. SELECT[col:​Customer_Orders:​customer_id] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Credit Card:​col:​Customers:​payment_method_code] <br>3. PROJECT[tbl:​Customer_Orders, #2] <br>4. GROUP[count, #3, #2] <br>5. COMPARATIVE[#2, #4, comparative:​>:​2] <br>6. PROJECT[col:​Customers:​customer_name, #5] <br> | + | hard | 
  | SPIDER_train_5666 | What are the name and phone of the customer with the most ordered product quantity? | SELECT  T1.customer_name ,  T1.customer_phone FROM customers AS T1 JOIN customer_orders AS T2 ON T1.customer_id  =  T2.customer_id JOIN order_items AS T3 ON T3.order_id  =  T2.order_id GROUP BY T1.customer_id ORDER BY sum(T3.order_quantity) DESC LIMIT 1 | 1. SELECT[tbl:​Customer_Orders] <br>2. PROJECT[tbl:​Order_Items, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​Customers:​customer_name, #4] <br>6. PROJECT[col:​Customers:​customer_phone, #4] <br>7. UNION[#5, #6] <br> | 1. SELECT[tbl:​Customer_Orders] <br>2. PROJECT[tbl:​Order_Items, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​Customers:​customer_name, #4] <br>6. PROJECT[col:​Customers:​customer_phone, #4] <br>7. UNION[#5, #6] <br> | + | extra | 
  | SPIDER_train_5667 | Show the product type and name for the products with price higher than 1000 or lower than 500. | SELECT product_type_code ,  product_name FROM products WHERE product_price  >  1000 OR product_price  <  500 | 1. SELECT[tbl:​Products] <br>2. PROJECT[col:​Products:​product_price, #1] <br>3. COMPARATIVE[#1, #2, comparative:​<:​500:​col:​Products:​product_price] <br>4. COMPARATIVE[#1, #2, comparative:​>:​1000:​col:​Products:​product_price] <br>5. UNION[#3, #4] <br>6. PROJECT[col:​Products:​product_type_code, #5] <br>7. PROJECT[col:​Products:​product_name, #5] <br>8. UNION[#6, #7] <br> | 1. SELECT[tbl:​Products] <br>2. PROJECT[col:​Products:​product_price, #1] <br>3. COMPARATIVE[#1, #2, comparative:​<:​500:​col:​Products:​product_price] <br>4. COMPARATIVE[#1, #2, comparative:​>:​1000:​col:​Products:​product_price] <br>5. UNION[#3, #4] <br>6. PROJECT[col:​Products:​product_type_code, #5] <br>7. PROJECT[col:​Products:​product_name, #5] <br>8. UNION[#6, #7] <br> | + | extra | 
 ***
 Exec acc: **0.9231**
