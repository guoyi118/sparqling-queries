 | Idx | Table      | Column | Primary Key | Foreign Key | 
 | ----------- | ----------- | ----------- | ----------- | ----------- | 
  | 0 |  | * |   |   | 
 | 1 | **Customers** | customer_id | + | --> 32 | 
 | 2 |   | customer_details |   |   | 
 | 3 | **Properties** | property_id | + | --> 8 | 
 | 4 |   | property_type_code |   |   | 
 | 5 |   | property_address |   |   | 
 | 6 |   | other_details |   |   | 
 | 7 | **Residents** | resident_id | + | --> 35 | 
 | 8 |   | property_id |   | --> 22 | 
 | 9 |   | date_moved_in |   | --> 21 | 
 | 10 |   | date_moved_out |   |   | 
 | 11 |   | other_details |   |   | 
 | 12 | **Organizations** | organization_id | + | --> 16 | 
 | 13 |   | parent_organization_id |   | --> 12 | 
 | 14 |   | organization_details |   |   | 
 | 15 | **Services** | service_id | + | --> 20 | 
 | 16 |   | organization_id |   |   | 
 | 17 |   | service_type_code |   |   | 
 | 18 |   | service_details |   |   | 
 | 19 | **Residents_Services** | resident_id | + | --> 7 | 
 | 20 |   | service_id |   |   | 
 | 21 |   | date_moved_in |   |   | 
 | 22 |   | property_id |   | --> 3 | 
 | 23 |   | date_requested |   |   | 
 | 24 |   | date_provided |   |   | 
 | 25 |   | other_details |   |   | 
 | 26 | **Things** | thing_id | + | --> 43 | 
 | 27 |   | organization_id |   |   | 
 | 28 |   | Type_of_Thing_Code |   |   | 
 | 29 |   | service_type_code |   |   | 
 | 30 |   | service_details |   |   | 
 | 31 | **Customer_Events** | Customer_Event_ID | + | --> 38 | 
 | 32 |   | customer_id |   |   | 
 | 33 |   | date_moved_in |   |   | 
 | 34 |   | property_id |   | --> 3 | 
 | 35 |   | resident_id |   | --> 7 | 
 | 36 |   | thing_id |   |   | 
 | 37 | **Customer_Event_Notes** | Customer_Event_Note_ID | + |   | 
 | 38 |   | Customer_Event_ID |   |   | 
 | 39 |   | service_type_code |   |   | 
 | 40 |   | resident_id |   | --> 7 | 
 | 41 |   | property_id |   | --> 3 | 
 | 42 |   | date_moved_in |   |   | 
 | 43 | **Timed_Status_of_Things** | thing_id | + |   | 
 | 44 |   | Date_and_Date |   |   | 
 | 45 |   | Status_of_Thing_Code |   |   | 
 | 46 | **Timed_Locations_of_Things** | thing_id | + |   | 
 | 47 |   | Date_and_Time |   |   | 
 | 48 |   | Location_Code |   |   | 
 
  | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |
  | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | 
 | SPIDER_train_4843 | How many residents does each property have? List property id and resident count. | SELECT T1.property_id ,  count(*) FROM properties AS T1 JOIN residents AS T2 ON T1.property_id  =  T2.property_id GROUP BY T1.property_id | 1. SELECT[col:​Properties:​property_id] <br>2. PROJECT[tbl:​Residents, #1] <br>3. GROUP[count, #2, #1] <br>4. PROJECT[col:​Properties:​property_id, #1] <br>5. UNION[#4, #3] <br> | 1. SELECT[col:​Properties:​property_id] <br>2. PROJECT[tbl:​Residents, #1] <br>3. GROUP[count, #2, #1] <br>4. PROJECT[col:​Properties:​property_id, #1] <br>5. UNION[#4, #3] <br> | + | medium | 
  | SPIDER_train_4844 | What is the distinct service types that are provided by the organization which has detail 'Denesik and Sons Party'? | SELECT DISTINCT T1.service_type_code FROM services AS T1 JOIN organizations AS T2 ON T1.organization_id  =  T2.organization_id WHERE T2.organization_details  =  'Denesik and Sons Party' | 1. SELECT[tbl:​Organizations] <br>2. PROJECT[col:​Organizations:​organization_details, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Denesik and Sons Party:​col:​Organizations:​organization_details] <br>4.*(distinct)* PROJECT[col:​Services:​service_type_code, #3] <br> | 1. SELECT[tbl:​Organizations] <br>2. PROJECT[col:​Organizations:​organization_details, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Denesik and Sons Party:​col:​Organizations:​organization_details] <br>4.*(distinct)* PROJECT[col:​Services:​service_type_code, #3] <br> | + | medium | 
  | SPIDER_train_4847 | List the id and type of each thing, and the details of the organization that owns it. | SELECT T1.thing_id ,  T1.type_of_Thing_Code ,  T2.organization_details FROM Things AS T1 JOIN Organizations AS T2 ON T1.organization_id  =  T2.organization_id | 1. SELECT[tbl:​Things] <br>2. PROJECT[col:​Things:​thing_id, #1] <br>3. PROJECT[col:​Things:​Type_of_Thing_Code, #1] <br>4. PROJECT[tbl:​Organizations, #1] <br>5. PROJECT[col:​Organizations:​organization_details, #4] <br>6. UNION[#2, #3, #5] <br> | 1. SELECT[tbl:​Things] <br>2. PROJECT[col:​Things:​thing_id, #1] <br>3. PROJECT[col:​Things:​Type_of_Thing_Code, #1] <br>4. PROJECT[tbl:​Organizations, #1] <br>5. PROJECT[col:​Organizations:​organization_details, #4] <br>6. UNION[#2, #3, #5] <br> | + | medium | 
  | SPIDER_train_4848 | What are the id and details of the customers who have at least 3 events? | SELECT T1.customer_id ,  T1.customer_details FROM Customers AS T1 JOIN Customer_Events AS T2 ON T1.customer_id  =  T2.customer_id GROUP BY T1.customer_id HAVING count(*)  >=  3 | 1. SELECT[tbl:​Customers] <br>2. PROJECT[tbl:​Customer_Events, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​3] <br>5. PROJECT[col:​Customers:​customer_id, #4] <br>6. PROJECT[col:​Customers:​customer_details, #4] <br>7. UNION[#5, #6] <br> | 1. SELECT[tbl:​Customers] <br>2. PROJECT[tbl:​Customer_Events, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​3] <br>5. PROJECT[col:​Customers:​customer_id, #4] <br>6. PROJECT[col:​Customers:​customer_details, #4] <br>7. UNION[#5, #6] <br> | + | medium | 
  | SPIDER_train_4849 | What is each customer's move in date, and the corresponding customer id and details? | SELECT T2.date_moved_in ,  T1.customer_id ,  T1.customer_details FROM Customers AS T1 JOIN Customer_Events AS T2 ON T1.customer_id  =  T2.customer_id | 1. SELECT[tbl:​Customers] <br>2. PROJECT[col:​Customer_Events:​date_moved_in, #1] <br>3. PROJECT[col:​Customers:​customer_id, #1] <br>4. PROJECT[col:​Customers:​customer_details, #1] <br>5. UNION[#2, #3, #4] <br> | 1. SELECT[tbl:​Customers] <br>2. PROJECT[col:​Customer_Events:​date_moved_in, #1] <br>3. PROJECT[col:​Customers:​customer_id, #1] <br>4. PROJECT[col:​Customers:​customer_details, #1] <br>5. UNION[#2, #3, #4] <br> | + | medium | 
  | SPIDER_train_4850 | Which events have the number of notes between one and three? List the event id and the property id. | SELECT T1.Customer_Event_ID  ,  T1.property_id FROM Customer_Events AS T1 JOIN Customer_Event_Notes AS T2 ON T1.Customer_Event_ID  =  T2.Customer_Event_ID GROUP BY T1.customer_event_id HAVING count(*) BETWEEN 1 AND 3 | 1. SELECT[tbl:​Customer_Events] <br>2. PROJECT[tbl:​Customer_Event_Notes, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​1] <br>5. COMPARATIVE[#1, #3, comparative:​>=:​1] <br>6. INTERSECTION[#1, #4, #5] <br>7. PROJECT[col:​Customer_Events:​Customer_Event_ID, #6] <br>8. PROJECT[col:​Customer_Events:​property_id, #6] <br>9. UNION[#7, #8] <br> | 1. SELECT[tbl:​Customer_Events] <br>2. PROJECT[tbl:​Customer_Event_Notes, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​1] <br>5. COMPARATIVE[#1, #3, comparative:​>=:​1] <br>6. INTERSECTION[#1, #4, #5] <br>7. PROJECT[col:​Customer_Events:​Customer_Event_ID, #6] <br>8. PROJECT[col:​Customer_Events:​property_id, #6] <br>9. UNION[#7, #8] <br> | + | medium | 
  | SPIDER_train_4852 | How many distinct locations have the things with service detail 'Unsatisfied' been located in? | SELECT count(DISTINCT T2.Location_Code) FROM Things AS T1 JOIN Timed_Locations_of_Things AS T2 ON T1.thing_id  =  T2.thing_id WHERE T1.service_details  =  'Unsatisfied' | 1. SELECT[col:​Timed_Locations_of_Things:​Location_Code] <br>2. PROJECT[tbl:​Things, #1] <br>3. PROJECT[col:​Things:​service_details, #2] <br>4. COMPARATIVE[#1, #3, comparative:​=:​Unsatisfied:​col:​Things:​service_details] <br>5.*(distinct)* PROJECT[distinct #REF, #4] <br>6. AGGREGATE[count, #5] <br> | 1. SELECT[col:​Timed_Locations_of_Things:​Location_Code] <br>2. PROJECT[tbl:​Things, #1] <br>3. PROJECT[col:​Things:​service_details, #2] <br>4. COMPARATIVE[#1, #3, comparative:​=:​Unsatisfied:​col:​Things:​service_details] <br>5.*(distinct)* PROJECT[None, #4] <br>6. AGGREGATE[count, #5] <br> | + | medium | 
  | SPIDER_train_4853 | How many different status codes of things are there? | SELECT count(DISTINCT Status_of_Thing_Code) FROM Timed_Status_of_Things | 1. SELECT[tbl:​Timed_Status_of_Things] <br>2. PROJECT[col:​Timed_Status_of_Things:​Status_of_Thing_Code, #1] <br>3.*(distinct)* PROJECT[different #REF, #2] <br>4. AGGREGATE[count, #3] <br> | 1. SELECT[tbl:​Timed_Status_of_Things] <br>2. PROJECT[col:​Timed_Status_of_Things:​Status_of_Thing_Code, #1] <br>3.*(distinct)* PROJECT[None, #2] <br>4. AGGREGATE[count, #3] <br> | + | easy | 
  | SPIDER_train_4856 | What are the resident details containing the substring 'Miss'? | SELECT other_details FROM Residents WHERE other_details LIKE '%Miss%' | 1. SELECT[tbl:​Residents] <br>2. PROJECT[tbl:​Residents, #1] <br>3. COMPARATIVE[#1, #2, comparative:​like:​%Miss%:​col:​Residents:​other_details] <br>4. PROJECT[col:​Residents:​other_details, #3] <br> | 1. SELECT[tbl:​Residents] <br>2. PROJECT[tbl:​Residents, #1] <br>3. COMPARATIVE[#1, #2, comparative:​like:​Miss:​col:​Residents:​other_details] <br>4. PROJECT[col:​Residents:​other_details, #3] <br> | + | medium | 
  | SPIDER_train_4857 | List the customer event id and the corresponding move in date and property id. | SELECT customer_event_id ,  date_moved_in ,  property_id FROM customer_events | 1. SELECT[col:​Customer_Events:​Customer_Event_ID] <br>2. PROJECT[col:​Customer_Events:​date_moved_in, #1] <br>3. PROJECT[col:​Customer_Events:​property_id, #1] <br>4. UNION[#1, #2, #3] <br> | 1. SELECT[col:​Customer_Events:​Customer_Event_ID] <br>2. PROJECT[col:​Customer_Events:​date_moved_in, #1] <br>3. PROJECT[col:​Customer_Events:​property_id, #1] <br>4. UNION[#1, #2, #3] <br> | + | medium | 
  | SPIDER_train_4858 | How many customers did not have any event? | SELECT count(*) FROM customers WHERE customer_id NOT IN ( SELECT customer_id FROM customer_events ) | 1. SELECT[tbl:​Customers] <br>2. FILTER[#1, tbl:​Customer_Events] <br>3. DISCARD[#1, #2] <br>4. AGGREGATE[count, #3] <br> | 1. SELECT[tbl:​Customers] <br>2. COMPARATIVE[#1, #1, tbl:​Customer_Events] <br>3. DISCARD[#1, #2] <br>4. AGGREGATE[count, #3] <br> | + | extra | 
  | SPIDER_train_4859 | What are the distinct move in dates of the residents? | SELECT DISTINCT date_moved_in FROM residents | 1. SELECT[tbl:​Residents] <br>2.*(distinct)* PROJECT[col:​Residents:​date_moved_in, #1] <br> | 1. SELECT[tbl:​Residents] <br>2.*(distinct)* PROJECT[col:​Residents:​date_moved_in, #1] <br> | + | easy | 
 ***
 Exec acc: **1.0000**
