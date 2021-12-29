 | Idx | Table      | Column | Primary Key | Foreign Key | 
 | ----------- | ----------- | ----------- | ----------- | ----------- | 
  | 0 |  | * |   |   | 
 | 1 | **Premises** | premise_id | + |   | 
 | 2 |   | premises_type |   |   | 
 | 3 |   | premise_details |   |   | 
 | 4 | **Products** | product_id | + |   | 
 | 5 |   | product_category |   |   | 
 | 6 |   | product_name |   |   | 
 | 7 | **Customers** | customer_id | + |   | 
 | 8 |   | payment_method |   |   | 
 | 9 |   | customer_name |   |   | 
 | 10 |   | customer_phone |   |   | 
 | 11 |   | customer_email |   |   | 
 | 12 |   | customer_address |   |   | 
 | 13 |   | customer_login |   |   | 
 | 14 |   | customer_password |   |   | 
 | 15 | **Mailshot_Campaigns** | mailshot_id | + |   | 
 | 16 |   | product_category |   |   | 
 | 17 |   | mailshot_name |   |   | 
 | 18 |   | mailshot_start_date |   |   | 
 | 19 |   | mailshot_end_date |   |   | 
 | 20 | **Customer_Addresses** | customer_id |   | --> 7 | 
 | 21 |   | premise_id |   | --> 1 | 
 | 22 |   | date_address_from |   |   | 
 | 23 |   | address_type_code |   |   | 
 | 24 |   | date_address_to |   |   | 
 | 25 | **Customer_Orders** | order_id | + |   | 
 | 26 |   | customer_id |   | --> 7 | 
 | 27 |   | order_status_code |   |   | 
 | 28 |   | shipping_method_code |   |   | 
 | 29 |   | order_placed_datetime |   |   | 
 | 30 |   | order_delivered_datetime |   |   | 
 | 31 |   | order_shipping_charges |   |   | 
 | 32 | **Mailshot_Customers** | mailshot_id |   | --> 15 | 
 | 33 |   | customer_id |   | --> 7 | 
 | 34 |   | outcome_code |   |   | 
 | 35 |   | mailshot_customer_date |   |   | 
 | 36 | **Order_Items** | item_id |   |   | 
 | 37 |   | order_item_status_code |   |   | 
 | 38 |   | order_id |   | --> 25 | 
 | 39 |   | product_id |   | --> 4 | 
 | 40 |   | item_status_code |   |   | 
 | 41 |   | item_delivered_datetime |   |   | 
 | 42 |   | item_order_quantity |   |   | 
 
  | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |
  | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | 
 | SPIDER_train_4620 | How many premises are there? | SELECT count(*) FROM premises | 1. SELECT[tbl:​Premises] <br>2. AGGREGATE[count, #1] <br> | 1. SELECT[tbl:​Premises] <br>2. AGGREGATE[count, #1] <br> | + | easy | 
  | SPIDER_train_4621 | What are all the distinct premise types? | SELECT DISTINCT premises_type FROM premises | 1. SELECT[tbl:​Premises] <br>2. PROJECT[col:​Premises:​premises_type, #1] <br>3.*(distinct)* PROJECT[distinct #REF, #2] <br> | 1. SELECT[tbl:​Premises] <br>2. PROJECT[col:​Premises:​premises_type, #1] <br>3.*(distinct)* PROJECT[None, #2] <br> | + | easy | 
  | SPIDER_train_4622 | Find the types and details for all premises and order by the premise type. | SELECT premises_type ,  premise_details FROM premises ORDER BY premises_type | 1. SELECT[tbl:​Premises] <br>2. PROJECT[col:​Premises:​premises_type, #1] <br>3. PROJECT[col:​Premises:​premise_details, #1] <br>4. UNION[#2, #3] <br>5. SORT[#4, #2] <br> | 1. SELECT[tbl:​Premises] <br>2. PROJECT[col:​Premises:​premises_type, #1] <br>3. PROJECT[col:​Premises:​premise_details, #1] <br>4. UNION[#2, #3] <br>5. SORT[#4, #2, sortdir:​ascending] <br> | + | medium | 
  | SPIDER_train_4623 | Show each premise type and the number of premises in that type. | SELECT premises_type ,  count(*) FROM premises GROUP BY premises_type | 1. SELECT[col:​Premises:​premises_type] <br>2. PROJECT[tbl:​Premises, #1] <br>3. GROUP[count, #2, #1] <br>4. UNION[#1, #3] <br> | 1. SELECT[col:​Premises:​premises_type] <br>2. PROJECT[tbl:​Premises, #1] <br>3. GROUP[count, #2, #1] <br>4. UNION[#1, #3] <br> | + | medium | 
  | SPIDER_train_4624 | Show all distinct product categories along with the number of mailshots in each category. | SELECT product_category ,  count(*) FROM mailshot_campaigns GROUP BY product_category | 1.*(distinct)* SELECT[col:​Mailshot_Campaigns:​product_category] <br>2. PROJECT[tbl:​Mailshot_Campaigns, #1] <br>3. GROUP[count, #2, #1] <br>4. UNION[#1, #3] <br> | 1.*(distinct)* SELECT[col:​Mailshot_Campaigns:​product_category] <br>2. PROJECT[tbl:​Mailshot_Campaigns, #1] <br>3. GROUP[count, #2, #1] <br>4. UNION[#1, #3] <br> | + | medium | 
  | SPIDER_train_4626 | Show the name and phone for customers with a mailshot with outcome code 'No Response'. | SELECT T1.customer_name ,  T1.customer_phone FROM customers AS T1 JOIN mailshot_customers AS T2 ON T1.customer_id  =  T2.customer_id WHERE T2.outcome_code  =  'No Response' | 1. SELECT[tbl:​Mailshot_Customers] <br>2. PROJECT[tbl:​Mailshot_Customers, #1] <br>3. PROJECT[col:​Mailshot_Customers:​outcome_code, #2] <br>4. COMPARATIVE[#1, #3, comparative:​=:​No Response:​col:​Mailshot_Customers:​outcome_code] <br>5. PROJECT[col:​Customers:​customer_name, #4] <br>6. PROJECT[col:​Customers:​customer_phone, #4] <br>7. UNION[#5, #6] <br> | 1. SELECT[tbl:​Mailshot_Customers] <br>2. PROJECT[tbl:​Mailshot_Customers, #1] <br>3. PROJECT[col:​Mailshot_Customers:​outcome_code, #2] <br>4. COMPARATIVE[#1, #3, comparative:​=:​No Response:​col:​Mailshot_Customers:​outcome_code] <br>5. PROJECT[col:​Customers:​customer_name, #4] <br>6. PROJECT[col:​Customers:​customer_phone, #4] <br>7. UNION[#5, #6] <br> | + | medium | 
  | SPIDER_train_4627 | Show the outcome code of mailshots along with the number of mailshots in each outcome code. | SELECT outcome_code ,  count(*) FROM mailshot_customers GROUP BY outcome_code | 1. SELECT[tbl:​Mailshot_Customers] <br>2. PROJECT[col:​Mailshot_Customers:​outcome_code, #1] <br>3. GROUP[count, #1, #2] <br>4. UNION[#2, #3] <br> | 1. SELECT[tbl:​Mailshot_Customers] <br>2. PROJECT[col:​Mailshot_Customers:​outcome_code, #1] <br>3. GROUP[count, #1, #2] <br>4. UNION[#2, #3] <br> | + | medium | 
  | SPIDER_train_4629 | Show the names of customers who have the most mailshots. | SELECT T2.customer_name FROM mailshot_customers AS T1 JOIN customers AS T2 ON T1.customer_id  =  T2.customer_id GROUP BY T1.customer_id ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[tbl:​Customers] <br>2. PROJECT[tbl:​Mailshot_Customers, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​Customers:​customer_name, #4] <br> | 1. SELECT[tbl:​Customers] <br>2. PROJECT[tbl:​Mailshot_Customers, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​Customers:​customer_name, #4] <br> | + | extra | 
  | SPIDER_train_4630 | What are the name and payment method of customers who have both mailshots in 'Order' outcome and mailshots in 'No Response' outcome. | SELECT T2.customer_name ,  T2.payment_method FROM mailshot_customers AS T1 JOIN customers AS T2 ON T1.customer_id  =  T2.customer_id WHERE T1.outcome_code  =  'Order' INTERSECT SELECT T2.customer_name ,  T2.payment_method FROM mailshot_customers AS T1 JOIN customers AS T2 ON T1.customer_id  =  T2.customer_id WHERE T1.outcome_code  =  'No Response' | 1. SELECT[tbl:​Customers] <br>2. PROJECT[tbl:​Mailshot_Customers, #1] <br>3. PROJECT[col:​Mailshot_Customers:​outcome_code, #2] <br>4. COMPARATIVE[#1, #3, comparative:​=:​Order:​col:​Mailshot_Customers:​outcome_code] <br>5. COMPARATIVE[#1, #3, comparative:​=:​No Response:​col:​Mailshot_Customers:​outcome_code] <br>6. INTERSECTION[#1, #4, #5] <br>7. PROJECT[col:​Customers:​customer_name, #6] <br>8. PROJECT[col:​Customers:​payment_method, #6] <br>9. UNION[#7, #8] <br> | 1. SELECT[tbl:​Customers] <br>2. PROJECT[tbl:​Mailshot_Customers, #1] <br>3. PROJECT[col:​Mailshot_Customers:​outcome_code, #2] <br>4. COMPARATIVE[#1, #3, comparative:​=:​Order:​col:​Mailshot_Customers:​outcome_code] <br>5. COMPARATIVE[#1, #3, comparative:​=:​No Response:​col:​Mailshot_Customers:​outcome_code] <br>6. INTERSECTION[#1, #4, #5] <br>7. PROJECT[col:​Customers:​customer_name, #6] <br>8. PROJECT[col:​Customers:​payment_method, #6] <br>9. UNION[#7, #8] <br> | + | extra | 
  | SPIDER_train_4631 | Show the premise type and address type code for all customer addresses. | SELECT T2.premises_type ,  T1.address_type_code FROM customer_addresses AS T1 JOIN premises AS T2 ON T1.premise_id  =  T2.premise_id | 1. SELECT[tbl:​Customer_Addresses] <br>2. PROJECT[tbl:​Customer_Addresses, #1] <br>3. PROJECT[col:​Premises:​premises_type, #2] <br>4. PROJECT[col:​Customer_Addresses:​address_type_code, #2] <br>5. UNION[#3, #4] <br> | 1. SELECT[tbl:​Customer_Addresses] <br>2. PROJECT[tbl:​Customer_Addresses, #1] <br>3. PROJECT[col:​Premises:​premises_type, #2] <br>4. PROJECT[col:​Customer_Addresses:​address_type_code, #2] <br>5. UNION[#3, #4] <br> | + | medium | 
  | SPIDER_train_4632 | What are the distinct address type codes for all customer addresses? | SELECT DISTINCT address_type_code FROM customer_addresses | 1. SELECT[tbl:​Customer_Addresses] <br>2. PROJECT[tbl:​Customer_Addresses, #1] <br>3.*(distinct)* PROJECT[col:​Customer_Addresses:​address_type_code, #2] <br> | 1. SELECT[tbl:​Customer_Addresses] <br>2. PROJECT[tbl:​Customer_Addresses, #1] <br>3.*(distinct)* PROJECT[col:​Customer_Addresses:​address_type_code, #2] <br> | + | easy | 
  | SPIDER_train_4633 | Show the shipping charge and customer id for customer orders with order status Cancelled or Paid. | SELECT order_shipping_charges ,  customer_id FROM customer_orders WHERE order_status_code  =  'Cancelled' OR order_status_code  =  'Paid' | 1. SELECT[tbl:​Customer_Orders] <br>2. PROJECT[tbl:​Customer_Orders, #1] <br>3. PROJECT[tbl:​Customer_Orders, #2] <br>4. COMPARATIVE[#2, #3, comparative:​=:​Cancelled:​col:​Customer_Orders:​order_status_code] <br>5. COMPARATIVE[#2, #3, comparative:​=:​Paid:​col:​Customer_Orders:​order_status_code] <br>6. UNION[#4, #5] <br>7. PROJECT[col:​Customer_Orders:​order_shipping_charges, #6] <br>8. PROJECT[col:​Customer_Orders:​customer_id, #6] <br>9. UNION[#7, #8] <br> | 1. SELECT[tbl:​Customer_Orders] <br>2. PROJECT[tbl:​Customer_Orders, #1] <br>3. PROJECT[tbl:​Customer_Orders, #2] <br>4. COMPARATIVE[#2, #3, comparative:​=:​Cancelled:​col:​Customer_Orders:​order_status_code] <br>5. COMPARATIVE[#2, #3, comparative:​=:​Paid:​col:​Customer_Orders:​order_status_code] <br>6. UNION[#4, #5] <br>7. PROJECT[col:​Customer_Orders:​order_shipping_charges, #6] <br>8. PROJECT[col:​Customer_Orders:​customer_id, #6] <br>9. UNION[#7, #8] <br> | + | extra | 
  | SPIDER_train_4634 | Show the names of customers having an order with shipping method FedEx and order status Paid. | SELECT T1.customer_name FROM customers AS T1 JOIN customer_orders AS T2 ON T1.customer_id  =  T2.customer_id WHERE shipping_method_code  =  'FedEx' AND order_status_code  =  'Paid' | 1. SELECT[tbl:​Customer_Orders] <br>2. PROJECT[tbl:​Customer_Orders, #1] <br>3. PROJECT[col:​Customer_Orders:​order_status_code, #2] <br>4. PROJECT[col:​Customer_Orders:​shipping_method_code, #2] <br>5. COMPARATIVE[#1, #3, comparative:​=:​Paid:​col:​Customer_Orders:​order_status_code] <br>6. COMPARATIVE[#1, #4, comparative:​=:​FedEx:​col:​Customer_Orders:​shipping_method_code] <br>7. INTERSECTION[#1, #5, #6] <br>8. PROJECT[col:​Customers:​customer_name, #7] <br> | 1. SELECT[tbl:​Customer_Orders] <br>2. PROJECT[tbl:​Customer_Orders, #1] <br>3. PROJECT[col:​Customer_Orders:​order_status_code, #2] <br>4. PROJECT[col:​Customer_Orders:​shipping_method_code, #2] <br>5. COMPARATIVE[#1, #3, comparative:​=:​Paid:​col:​Customer_Orders:​order_status_code] <br>6. COMPARATIVE[#1, #4, comparative:​=:​FedEx:​col:​Customer_Orders:​shipping_method_code] <br>7. INTERSECTION[#1, #5, #6] <br>8. PROJECT[col:​Customers:​customer_name, #7] <br> | + | medium | 
 ***
 Exec acc: **1.0000**
