 | Idx | Table      | Column | Primary Key | Foreign Key | 
 | ----------- | ----------- | ----------- | ----------- | ----------- | 
  | 0 |  | * |   |   | 
 | 1 | **Products** | product_id | + |   | 
 | 2 |   | product_name |   |   | 
 | 3 |   | product_price |   |   | 
 | 4 |   | product_description |   |   | 
 | 5 | **Addresses** | address_id | + |   | 
 | 6 |   | address_details |   |   | 
 | 7 |   | city |   |   | 
 | 8 |   | zip_postcode |   |   | 
 | 9 |   | state_province_county |   |   | 
 | 10 |   | country |   |   | 
 | 11 | **Customers** | customer_id | + |   | 
 | 12 |   | payment_method |   |   | 
 | 13 |   | customer_name |   |   | 
 | 14 |   | customer_phone |   |   | 
 | 15 |   | customer_email |   |   | 
 | 16 |   | date_became_customer |   |   | 
 | 17 | **Regular_Orders** | regular_order_id | + |   | 
 | 18 |   | distributer_id |   | --> 11 | 
 | 19 | **Regular_Order_Products** | regular_order_id |   | --> 17 | 
 | 20 |   | product_id |   | --> 1 | 
 | 21 | **Actual_Orders** | actual_order_id | + |   | 
 | 22 |   | order_status_code |   |   | 
 | 23 |   | regular_order_id |   | --> 17 | 
 | 24 |   | actual_order_date |   |   | 
 | 25 | **Actual_Order_Products** | actual_order_id |   | --> 21 | 
 | 26 |   | product_id |   | --> 1 | 
 | 27 | **Customer_Addresses** | customer_id |   | --> 11 | 
 | 28 |   | address_id |   | --> 5 | 
 | 29 |   | date_from |   |   | 
 | 30 |   | address_type |   |   | 
 | 31 |   | date_to |   |   | 
 | 32 | **Delivery_Routes** | route_id | + |   | 
 | 33 |   | route_name |   |   | 
 | 34 |   | other_route_details |   |   | 
 | 35 | **Delivery_Route_Locations** | location_code | + |   | 
 | 36 |   | route_id |   | --> 32 | 
 | 37 |   | location_address_id |   | --> 5 | 
 | 38 |   | location_name |   |   | 
 | 39 | **Trucks** | truck_id | + |   | 
 | 40 |   | truck_licence_number |   |   | 
 | 41 |   | truck_details |   |   | 
 | 42 | **Employees** | employee_id | + |   | 
 | 43 |   | employee_address_id |   | --> 5 | 
 | 44 |   | employee_name |   |   | 
 | 45 |   | employee_phone |   |   | 
 | 46 | **Order_Deliveries** | location_code |   | --> 35 | 
 | 47 |   | actual_order_id |   | --> 21 | 
 | 48 |   | delivery_status_code |   |   | 
 | 49 |   | driver_employee_id |   | --> 42 | 
 | 50 |   | truck_id |   | --> 39 | 
 | 51 |   | delivery_date |   |   | 
 
  | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |
  | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | 
 | SPIDER_train_2842 | Find the ids of orders whose status is 'Success'. | SELECT actual_order_id FROM actual_orders WHERE order_status_code  =  'Success' | 1. SELECT[tbl:​Actual_Orders] <br>2. PROJECT[col:​Actual_Orders:​order_status_code, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Success:​col:​Actual_Orders:​order_status_code] <br>4. PROJECT[col:​Actual_Orders:​actual_order_id, #3] <br> | 1. SELECT[tbl:​Actual_Orders] <br>2. PROJECT[col:​Actual_Orders:​order_status_code, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Success:​col:​Actual_Orders:​order_status_code] <br>4. PROJECT[col:​Actual_Orders:​actual_order_id, #3] <br> | + | easy | 
  | SPIDER_train_2843 | Find the name and price of the product that has been ordered the greatest number of times. | SELECT t1.product_name ,   t1.product_price FROM products AS t1 JOIN regular_order_products AS t2 ON t1.product_id  =  t2.product_id GROUP BY t2.product_id ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[tbl:​Products] <br>2. PROJECT[tbl:​Regular_Order_Products, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​max:​None] <br>5. PROJECT[col:​Products:​product_name, #4] <br>6. PROJECT[col:​Products:​product_price, #4] <br>7. UNION[#5, #6] <br> | 1. SELECT[tbl:​Products] <br>2. PROJECT[tbl:​Regular_Order_Products, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​Products:​product_name, #4] <br>6. PROJECT[col:​Products:​product_price, #4] <br>7. UNION[#5, #6] <br> | + | extra | 
  | SPIDER_train_2844 | Find the number of customers in total. | SELECT count(*) FROM customers | 1. SELECT[tbl:​Customers] <br>2. AGGREGATE[count, #1] <br> | 1. SELECT[tbl:​Customers] <br>2. AGGREGATE[count, #1] <br> | + | easy | 
  | SPIDER_train_2845 | How many different payment methods are there? | SELECT count(DISTINCT payment_method) FROM customers | 1. SELECT[col:​Customers:​payment_method] <br>2.*(distinct)* PROJECT[different #REF, #1] <br>3. AGGREGATE[count, #2] <br> | 1. SELECT[col:​Customers:​payment_method] <br>2.*(distinct)* PROJECT[None, #1] <br>3. AGGREGATE[count, #2] <br> | + | easy | 
  | SPIDER_train_2846 | Show the details of all trucks in the order of their license number. | SELECT truck_details FROM trucks ORDER BY truck_licence_number | 1. SELECT[tbl:​Trucks] <br>2. PROJECT[col:​Trucks:​truck_details, #1] <br>3. PROJECT[col:​Trucks:​truck_licence_number, #1] <br>4. SORT[#2, #3] <br> | 1. SELECT[tbl:​Trucks] <br>2. PROJECT[col:​Trucks:​truck_details, #1] <br>3. PROJECT[col:​Trucks:​truck_licence_number, #1] <br>4. SORT[#2, #3, sortdir:​ascending] <br> | + | easy | 
  | SPIDER_train_2849 | List the names and emails of customers who payed by Visa card. | SELECT customer_email ,  customer_name FROM customers WHERE payment_method  =  'Visa' | 1. SELECT[tbl:​Customers] <br>2. FILTER[#1, comparative:​=:​Visa:​col:​Customers:​payment_method] <br>3. PROJECT[col:​Customers:​customer_email, #2] <br>4. PROJECT[col:​Customers:​customer_name, #2] <br>5. UNION[#3, #4] <br> | 1. SELECT[tbl:​Customers] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Visa:​col:​Customers:​payment_method] <br>3. PROJECT[col:​Customers:​customer_email, #2] <br>4. PROJECT[col:​Customers:​customer_name, #2] <br>5. UNION[#3, #4] <br> | + | medium | 
  | SPIDER_train_2850 | Find the names and phone numbers of customers living in California state. | SELECT t1.customer_name ,  t1.customer_phone FROM customers AS t1 JOIN customer_addresses AS t2 ON t1.customer_id  =  t2.customer_id JOIN addresses AS t3 ON t2.address_id  =  t3.address_id WHERE t3.state_province_county  =  'California' | 1. SELECT[tbl:​Customers] <br>2. FILTER[#1, comparative:​=:​California:​col:​Addresses:​state_province_county] <br>3. PROJECT[col:​Customers:​customer_name, #2] <br>4. PROJECT[col:​Customers:​customer_phone, #2] <br>5. UNION[#3, #4] <br> | 1. SELECT[tbl:​Customers] <br>2. COMPARATIVE[#1, #1, comparative:​=:​California:​col:​Addresses:​state_province_county] <br>3. PROJECT[col:​Customers:​customer_name, #2] <br>4. PROJECT[col:​Customers:​customer_phone, #2] <br>5. UNION[#3, #4] <br> | + | hard | 
  | SPIDER_train_2852 | List the names, phone numbers, and emails of all customers sorted by their dates of becoming customers. | SELECT customer_name ,  customer_phone ,  customer_email FROM Customers ORDER BY date_became_customer | 1. SELECT[tbl:​Customers] <br>2. PROJECT[col:​Customers:​customer_name, #1] <br>3. PROJECT[col:​Customers:​customer_phone, #1] <br>4. PROJECT[col:​Customers:​customer_email, #1] <br>5. PROJECT[col:​Customers:​date_became_customer, #1] <br>6. UNION[#2, #3, #4] <br>7. SORT[#6, #5] <br> | 1. SELECT[tbl:​Customers] <br>2. PROJECT[col:​Customers:​customer_name, #1] <br>3. PROJECT[col:​Customers:​customer_phone, #1] <br>4. PROJECT[col:​Customers:​customer_email, #1] <br>5. PROJECT[col:​Customers:​date_became_customer, #1] <br>6. UNION[#2, #3, #4] <br>7. SORT[#6, #5, sortdir:​ascending] <br> | + | medium | 
  | SPIDER_train_2855 | List the names of all routes in alphabetic order. | SELECT route_name FROM Delivery_Routes ORDER BY route_name | 1. SELECT[tbl:​Delivery_Routes] <br>2. PROJECT[col:​Delivery_Routes:​route_name, #1] <br>3. SORT[#2, #2, sortdir:​ascending] <br> | 1. SELECT[tbl:​Delivery_Routes] <br>2. PROJECT[col:​Delivery_Routes:​route_name, #1] <br>3. SORT[#2, #2, sortdir:​ascending] <br> | + | easy | 
  | SPIDER_train_2856 | Find the name of route that has the highest number of deliveries. | SELECT t1.route_name FROM Delivery_Routes AS t1 JOIN Delivery_Route_Locations AS t2 ON t1.route_id  =  t2.route_id GROUP BY t1.route_id ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[tbl:​Delivery_Routes] <br>2. PROJECT[tbl:​Delivery_Route_Locations, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​Delivery_Routes:​route_name, #4] <br> | 1. SELECT[tbl:​Delivery_Routes] <br>2. PROJECT[tbl:​Delivery_Route_Locations, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​Delivery_Routes:​route_name, #4] <br> | + | extra | 
  | SPIDER_train_2857 | List the state names and the number of customers living in each state. | SELECT t2.state_province_county ,  count(*) FROM customer_addresses AS t1 JOIN addresses AS t2 ON t1.address_id  =  t2.address_id GROUP BY t2.state_province_county | 1. SELECT[col:​Addresses:​state_province_county] <br>2. PROJECT[col:​Addresses:​state_province_county, #1] <br>3. PROJECT[tbl:​Customer_Addresses, #1] <br>4. GROUP[count, #3, #1] <br>5. UNION[#2, #4] <br> | 1. SELECT[col:​Addresses:​state_province_county] <br>2. PROJECT[col:​Addresses:​state_province_county, #1] <br>3. PROJECT[tbl:​Customer_Addresses, #1] <br>4. GROUP[count, #3, #1] <br>5. UNION[#2, #4] <br> | + | medium | 
 ***
 Exec acc: **1.0000**
