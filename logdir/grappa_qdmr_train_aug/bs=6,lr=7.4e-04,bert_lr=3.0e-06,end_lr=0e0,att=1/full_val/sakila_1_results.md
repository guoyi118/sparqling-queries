 | Idx | Table      | Column | Primary Key | Foreign Key | 
 | ----------- | ----------- | ----------- | ----------- | ----------- | 
  | 0 |  | * |   |   | 
 | 1 | **actor** | actor_id | + | --> 45 | 
 | 2 |   | first_name |   |   | 
 | 3 |   | last_name |   |   | 
 | 4 |   | last_update |   |   | 
 | 5 | **address** | address_id | + | --> 28 | 
 | 6 |   | address |   |   | 
 | 7 |   | address2 |   |   | 
 | 8 |   | district |   |   | 
 | 9 |   | city_id |   | --> 16 | 
 | 10 |   | postal_code |   |   | 
 | 11 |   | phone |   |   | 
 | 12 |   | last_update |   |   | 
 | 13 | **category** | category_id | + | --> 49 | 
 | 14 |   | name |   |   | 
 | 15 |   | last_update |   |   | 
 | 16 | **city** | city_id | + |   | 
 | 17 |   | city |   |   | 
 | 18 |   | country_id |   | --> 20 | 
 | 19 |   | last_update |   |   | 
 | 20 | **country** | country_id | + |   | 
 | 21 |   | country |   |   | 
 | 22 |   | last_update |   |   | 
 | 23 | **customer** | customer_id | + | --> 71 | 
 | 24 |   | store_id |   | --> 86 | 
 | 25 |   | first_name |   |   | 
 | 26 |   | last_name |   |   | 
 | 27 |   | email |   |   | 
 | 28 |   | address_id |   |   | 
 | 29 |   | active |   |   | 
 | 30 |   | create_date |   |   | 
 | 31 |   | last_update |   |   | 
 | 32 | **film** | film_id | + | --> 48 | 
 | 33 |   | title |   |   | 
 | 34 |   | description |   |   | 
 | 35 |   | release_year |   |   | 
 | 36 |   | language_id |   | --> 58 | 
 | 37 |   | original_language_id |   | --> 58 | 
 | 38 |   | rental_duration |   |   | 
 | 39 |   | rental_rate |   |   | 
 | 40 |   | length |   |   | 
 | 41 |   | replacement_cost |   |   | 
 | 42 |   | rating |   |   | 
 | 43 |   | special_features |   |   | 
 | 44 |   | last_update |   |   | 
 | 45 | **film_actor** | actor_id | + |   | 
 | 46 |   | film_id |   |   | 
 | 47 |   | last_update |   |   | 
 | 48 | **film_category** | film_id | + |   | 
 | 49 |   | category_id |   |   | 
 | 50 |   | last_update |   |   | 
 | 51 | **film_text** | film_id | + | --> 32 | 
 | 52 |   | title |   |   | 
 | 53 |   | description |   |   | 
 | 54 | **inventory** | inventory_id | + | --> 70 | 
 | 55 |   | film_id |   |   | 
 | 56 |   | store_id |   | --> 86 | 
 | 57 |   | last_update |   |   | 
 | 58 | **language** | language_id | + |   | 
 | 59 |   | name |   |   | 
 | 60 |   | last_update |   |   | 
 | 61 | **payment** | payment_id | + |   | 
 | 62 |   | customer_id |   |   | 
 | 63 |   | staff_id |   | --> 75 | 
 | 64 |   | rental_id |   | --> 68 | 
 | 65 |   | amount |   |   | 
 | 66 |   | payment_date |   |   | 
 | 67 |   | last_update |   |   | 
 | 68 | **rental** | rental_id | + |   | 
 | 69 |   | rental_date |   |   | 
 | 70 |   | inventory_id |   |   | 
 | 71 |   | customer_id |   |   | 
 | 72 |   | return_date |   |   | 
 | 73 |   | staff_id |   | --> 75 | 
 | 74 |   | last_update |   |   | 
 | 75 | **staff** | staff_id | + | --> 87 | 
 | 76 |   | first_name |   |   | 
 | 77 |   | last_name |   |   | 
 | 78 |   | address_id |   |   | 
 | 79 |   | picture |   |   | 
 | 80 |   | email |   |   | 
 | 81 |   | store_id |   | --> 86 | 
 | 82 |   | active |   |   | 
 | 83 |   | username |   |   | 
 | 84 |   | password |   |   | 
 | 85 |   | last_update |   |   | 
 | 86 | **store** | store_id | + |   | 
 | 87 |   | manager_staff_id |   |   | 
 | 88 |   | address_id |   |   | 
 | 89 |   | last_update |   |   | 
 
  | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |
  | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | 
 | SPIDER_train_2925 | Count the number of different last names actors have. | SELECT count(DISTINCT last_name) FROM actor | 1. SELECT[tbl:​actor] <br>2.*(distinct)* PROJECT[col:​actor:​last_name, #1] <br>3. AGGREGATE[count, #2] <br> | 1. SELECT[tbl:​actor] <br>2.*(distinct)* PROJECT[col:​actor:​last_name, #1] <br>3. AGGREGATE[count, #2] <br> | + | easy | 
  | SPIDER_train_2926 | What is the most popular first name of the actors? | SELECT first_name FROM actor GROUP BY first_name ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[tbl:​actor] <br>2. PROJECT[col:​actor:​first_name, #1] <br>3. GROUP[count, #1, #2] <br>4. SUPERLATIVE[comparative:​max:​None, #2, #3] <br> | 1. SELECT[tbl:​actor] <br>2. PROJECT[col:​actor:​first_name, #1] <br>3. GROUP[count, #1, #2] <br>4. SUPERLATIVE[comparative:​max:​None, #2, #3] <br> | + | hard | 
  | SPIDER_train_2927 | Return the most common first name among all actors. | SELECT first_name FROM actor GROUP BY first_name ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[col:​actor:​first_name] <br>2. PROJECT[tbl:​actor, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | 1. SELECT[col:​actor:​first_name] <br>2. PROJECT[tbl:​actor, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | + | hard | 
  | SPIDER_train_2930 | Which districts have at least two addresses? | SELECT district FROM address GROUP BY district HAVING count(*)  >=  2 | 1. SELECT[col:​address:​district] <br>2. PROJECT[tbl:​address, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​2] <br> | 1. SELECT[col:​address:​district] <br>2. PROJECT[tbl:​address, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​2] <br> | + | easy | 
  | SPIDER_train_2931 | Give the districts which have two or more addresses. | SELECT district FROM address GROUP BY district HAVING count(*)  >=  2 | 1. SELECT[col:​address:​district] <br>2. PROJECT[tbl:​address, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​2] <br> | 1. SELECT[col:​address:​district] <br>2. PROJECT[tbl:​address, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​2] <br> | + | easy | 
  | SPIDER_train_2933 | Give the phone and postal code corresponding to the address '1031 Daugavpils Parkway'. | SELECT phone ,  postal_code FROM address WHERE address  =  '1031 Daugavpils Parkway' | 1. SELECT[tbl:​address] <br>2. PROJECT[col:​address:​phone, #1] <br>3. PROJECT[col:​address:​postal_code, #1] <br>4. UNION[#2, #3] <br>5. COMPARATIVE[#4, #1, comparative:​=:​1031 Daugavpils Parkway:​col:​address:​address] <br> | 1. SELECT[tbl:​address] <br>2. PROJECT[col:​address:​phone, #1] <br>3. PROJECT[col:​address:​postal_code, #1] <br>4. UNION[#2, #3] <br>5. COMPARATIVE[#4, #1, comparative:​=:​1031 Daugavpils Parkway:​col:​address:​address] <br> | + | medium | 
  | SPIDER_train_2934 | Which city has the most addresses? List the city name, number of addresses, and city id. | SELECT T2.city ,  count(*) ,  T1.city_id FROM address AS T1 JOIN city AS T2 ON T1.city_id  =  T2.city_id GROUP BY T1.city_id ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[col:​address:​city_id] <br>2. PROJECT[tbl:​address, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​max:​None] <br>5. PROJECT[col:​city:​city, #4] <br>6. PROJECT[col:​address:​city_id, #4] <br>7. AGGREGATE[max, #3] <br>8. UNION[#5, #7, #6] <br> | 1. SELECT[col:​address:​city_id] <br>2. PROJECT[tbl:​address, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​city:​city, #4] <br>6. PROJECT[col:​address:​city_id, #4] <br>7. AGGREGATE[max, #3] <br>8. UNION[#5, #7, #6] <br> | + | extra | 
  | SPIDER_train_2936 | How many addresses are in the district of California? | SELECT count(*) FROM address WHERE district  =  'California' | 1. SELECT[col:​address:​district] <br>2. PROJECT[tbl:​address, #1] <br>3. COMPARATIVE[#2, #1, comparative:​=:​California:​col:​address:​district] <br>4. AGGREGATE[count, #3] <br> | 1. SELECT[col:​address:​district] <br>2. PROJECT[tbl:​address, #1] <br>3. COMPARATIVE[#2, #1, comparative:​=:​California:​col:​address:​district] <br>4. AGGREGATE[count, #3] <br> | + | easy | 
  | SPIDER_train_2937 | Count the number of addressed in the California district. | SELECT count(*) FROM address WHERE district  =  'California' | 1. SELECT[col:​address:​district] <br>2. FILTER[#1, comparative:​=:​California:​col:​address:​district] <br>3. PROJECT[tbl:​address, #2] <br>4. AGGREGATE[count, #3] <br> | 1. SELECT[col:​address:​district] <br>2. COMPARATIVE[#1, #1, comparative:​=:​California:​col:​address:​district] <br>3. PROJECT[tbl:​address, #2] <br>4. AGGREGATE[count, #3] <br> | + | easy | 
  | SPIDER_train_2940 | How many cities are in Australia? | SELECT count(*) FROM city AS T1 JOIN country AS T2 ON T1.country_id  =  T2.country_id WHERE T2.country  =  'Australia' | 1. SELECT[val:​country:​country:​Australia] <br>2. PROJECT[tbl:​city, #1] <br>3. AGGREGATE[count, #2] <br> | 1. SELECT[val:​country:​country:​Australia] <br>2. PROJECT[tbl:​city, #1] <br>3. AGGREGATE[count, #2] <br> | + | medium | 
  | SPIDER_train_2941 | Count the number of cities in Australia. | SELECT count(*) FROM city AS T1 JOIN country AS T2 ON T1.country_id  =  T2.country_id WHERE T2.country  =  'Australia' | 1. SELECT[tbl:​city] <br>2. FILTER[#1, comparative:​=:​Australia:​col:​country:​country] <br>3. AGGREGATE[count, #2] <br> | 1. SELECT[tbl:​city] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Australia:​col:​country:​country] <br>3. AGGREGATE[count, #2] <br> | + | medium | 
  | SPIDER_train_2942 | Which countries have at least 3 cities? | SELECT T2.country FROM city AS T1 JOIN country AS T2 ON T1.country_id  =  T2.country_id GROUP BY T2.country_id HAVING count(*)  >=  3 | 1. SELECT[col:​country:​country] <br>2. PROJECT[tbl:​city, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​3] <br> | 1. SELECT[col:​country:​country] <br>2. PROJECT[tbl:​city, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​3] <br> | + | medium | 
  | SPIDER_train_2943 | What are the countries that contain 3 or more cities? | SELECT T2.country FROM city AS T1 JOIN country AS T2 ON T1.country_id  =  T2.country_id GROUP BY T2.country_id HAVING count(*)  >=  3 | 1. SELECT[col:​country:​country] <br>2. PROJECT[tbl:​city, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​3] <br> | 1. SELECT[col:​country:​country] <br>2. PROJECT[tbl:​city, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​3] <br> | + | medium | 
  | SPIDER_train_2946 | How many customers have an active value of 1? | SELECT count(*) FROM customer WHERE active = '1' | 1. SELECT[tbl:​customer] <br>2. PROJECT[col:​customer:​active, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​1:​col:​customer:​active] <br>4. AGGREGATE[count, #3] <br> | 1. SELECT[tbl:​customer] <br>2. PROJECT[col:​customer:​active, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​1:​col:​customer:​active] <br>4. AGGREGATE[count, #3] <br> | - | easy | 
  | SPIDER_train_2948 | Which film has the highest rental rate? And what is the rate? | SELECT title ,  rental_rate FROM film ORDER BY rental_rate DESC LIMIT 1 | 1. SELECT[col:​film:​title] <br>2. PROJECT[col:​film:​rental_rate, #1] <br>3. SUPERLATIVE[comparative:​max:​None, #1, #2] <br>4. AGGREGATE[max, #2] <br>5. UNION[#3, #4] <br> | 1. SELECT[col:​film:​title] <br>2. PROJECT[col:​film:​rental_rate, #1] <br>3. SUPERLATIVE[comparative:​max:​None, #1, #2] <br>4. AGGREGATE[max, #2] <br>5. UNION[#3, #4] <br> | + | medium | 
  | SPIDER_train_2949 | What are the title and rental rate of the film with the highest rental rate? | SELECT title ,  rental_rate FROM film ORDER BY rental_rate DESC LIMIT 1 | 1. SELECT[tbl:​film] <br>2. PROJECT[col:​film:​rental_rate, #1] <br>3. SUPERLATIVE[comparative:​max:​None, #1, #2] <br>4. PROJECT[col:​film:​title, #3] <br>5. PROJECT[col:​film:​rental_rate, #3] <br>6. UNION[#4, #5] <br> | 1. SELECT[tbl:​film] <br>2. PROJECT[col:​film:​rental_rate, #1] <br>3. SUPERLATIVE[comparative:​max:​None, #1, #2] <br>4. PROJECT[col:​film:​title, #3] <br>5. PROJECT[col:​film:​rental_rate, #3] <br>6. UNION[#4, #5] <br> | + | medium | 
  | SPIDER_train_2956 | Which store owns most items? | SELECT store_id FROM inventory GROUP BY store_id ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[col:​inventory:​store_id] <br>2. PROJECT[items #REF owns, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | 1. SELECT[col:​inventory:​store_id] <br>2. PROJECT[None, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | + | hard | 
  | SPIDER_train_2957 | What is the id of the store that has the most items in inventory? | SELECT store_id FROM inventory GROUP BY store_id ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[col:​inventory:​store_id] <br>2. PROJECT[tbl:​inventory, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​inventory:​store_id, #4] <br> | 1. SELECT[col:​inventory:​store_id] <br>2. PROJECT[tbl:​inventory, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​inventory:​store_id, #4] <br> | + | hard | 
  | SPIDER_train_2958 | What is the total amount of all payments? | SELECT sum(amount) FROM payment | 1. SELECT[tbl:​payment] <br>2. PROJECT[col:​payment:​amount, #1] <br>3. AGGREGATE[sum, #2] <br> | 1. SELECT[tbl:​payment] <br>2. PROJECT[col:​payment:​amount, #1] <br>3. AGGREGATE[sum, #2] <br> | + | easy | 
  | SPIDER_train_2959 | Return the sum of all payment amounts. | SELECT sum(amount) FROM payment | 1. SELECT[tbl:​payment] <br>2. PROJECT[col:​payment:​amount, #1] <br>3. AGGREGATE[sum, #2] <br> | 1. SELECT[tbl:​payment] <br>2. PROJECT[col:​payment:​amount, #1] <br>3. AGGREGATE[sum, #2] <br> | + | easy | 
  | SPIDER_train_2962 | What is the genre name of the film HUNGER ROOF? | SELECT T1.name FROM category AS T1 JOIN film_category AS T2 ON T1.category_id  =  T2.category_id JOIN film AS T3 ON T2.film_id  =  T3.film_id WHERE T3.title  =  'HUNGER ROOF' | 1. SELECT[tbl:​film_category] <br>2. PROJECT[genre of #REF, #1] <br>3. COMPARATIVE[#2, #1, comparative:​=:​HUNGER ROOF:​col:​film:​title] <br>4. PROJECT[col:​category:​name, #3] <br> | 1. SELECT[tbl:​film_category] <br>2. PROJECT[None, #1] <br>3. COMPARATIVE[#2, #1, comparative:​=:​HUNGER ROOF:​col:​film:​title] <br>4. PROJECT[col:​category:​name, #3] <br> | + | hard | 
  | SPIDER_train_2963 | Return the name of the category to which the film 'HUNGER ROOF' belongs. | SELECT T1.name FROM category AS T1 JOIN film_category AS T2 ON T1.category_id  =  T2.category_id JOIN film AS T3 ON T2.film_id  =  T3.film_id WHERE T3.title  =  'HUNGER ROOF' | 1. SELECT[val:​film:​title:​HUNGER ROOF] <br>2. PROJECT[tbl:​film_category, #1] <br>3. PROJECT[col:​category:​name, #2] <br> | 1. SELECT[val:​film:​title:​HUNGER ROOF] <br>2. PROJECT[tbl:​film_category, #1] <br>3. PROJECT[col:​category:​name, #2] <br> | + | hard | 
  | SPIDER_train_2964 | How many films are there in each category? List the genre name, genre id and the count. | SELECT T2.name ,  T1.category_id ,  count(*) FROM film_category AS T1 JOIN category AS T2 ON T1.category_id  =  T2.category_id GROUP BY T1.category_id | 1. SELECT[col:​film_category:​category_id] <br>2. PROJECT[tbl:​film_category, #1] <br>3. PROJECT[col:​category:​name, #1] <br>4. PROJECT[col:​film_category:​category_id, #1] <br>5. GROUP[count, #2, #1] <br>6. UNION[#3, #4, #5] <br> | 1. SELECT[col:​film_category:​category_id] <br>2. PROJECT[tbl:​film_category, #1] <br>3. PROJECT[col:​category:​name, #1] <br>4. PROJECT[col:​film_category:​category_id, #1] <br>5. GROUP[count, #2, #1] <br>6. UNION[#3, #4, #5] <br> | + | medium | 
  | SPIDER_train_2965 | What are the names and ids of the different categories, and how many films are in each? | SELECT T2.name ,  T1.category_id ,  count(*) FROM film_category AS T1 JOIN category AS T2 ON T1.category_id  =  T2.category_id GROUP BY T1.category_id | 1.*(distinct)* SELECT[col:​film_category:​category_id] <br>2. PROJECT[col:​category:​name, #1] <br>3. PROJECT[col:​film_category:​category_id, #1] <br>4. PROJECT[tbl:​film_category, #1] <br>5. GROUP[count, #4, #1] <br>6. UNION[#2, #3, #5] <br> | 1.*(distinct)* SELECT[col:​film_category:​category_id] <br>2. PROJECT[col:​category:​name, #1] <br>3. PROJECT[col:​film_category:​category_id, #1] <br>4. PROJECT[tbl:​film_category, #1] <br>5. GROUP[count, #4, #1] <br>6. UNION[#2, #3, #5] <br> | + | medium | 
  | SPIDER_train_2966 | Which film has the most copies in the inventory? List both title and id. | SELECT T1.title ,  T1.film_id FROM film AS T1 JOIN inventory AS T2 ON T1.film_id  =  T2.film_id GROUP BY T1.film_id ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[col:​inventory:​film_id] <br>2. PROJECT[tbl:​inventory, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​film:​title, #4] <br>6. PROJECT[col:​film:​film_id, #4] <br>7. UNION[#5, #6] <br> | 1. SELECT[col:​inventory:​film_id] <br>2. PROJECT[tbl:​inventory, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​film:​title, #4] <br>6. PROJECT[col:​film:​film_id, #4] <br>7. UNION[#5, #6] <br> | + | extra | 
  | SPIDER_train_2971 | Count the number of different languages in these films. | SELECT count(DISTINCT language_id) FROM film | 1. SELECT[tbl:​film] <br>2.*(distinct)* PROJECT[col:​film:​language_id, #1] <br>3. AGGREGATE[count, #2] <br> | 1. SELECT[tbl:​film] <br>2.*(distinct)* PROJECT[col:​film:​language_id, #1] <br>3. AGGREGATE[count, #2] <br> | + | easy | 
  | SPIDER_train_2974 | Where is store 1 located? | SELECT T2.address FROM store AS T1 JOIN address AS T2 ON T1.address_id  =  T2.address_id WHERE store_id  =  1 | 1. SELECT[val:​store:​store_id:​1] <br>2. PROJECT[col:​address:​address, #1] <br> | 1. SELECT[val:​rental:​inventory_id:​1] <br>2. PROJECT[col:​address:​address, #1] <br> | - | medium | 
  | SPIDER_train_2975 | Return the address of store 1. | SELECT T2.address FROM store AS T1 JOIN address AS T2 ON T1.address_id  =  T2.address_id WHERE store_id  =  1 | 1. SELECT[val:​store:​store_id:​1] <br>2. PROJECT[col:​address:​address, #1] <br> | 1. SELECT[val:​rental:​inventory_id:​1] <br>2. PROJECT[col:​address:​address, #1] <br> | - | medium | 
  | SPIDER_train_2978 | Which language does the film AIRPORT POLLOCK use? List the language name. | SELECT T2.name FROM film AS T1 JOIN LANGUAGE AS T2 ON T1.language_id  =  T2.language_id WHERE T1.title  =  'AIRPORT POLLOCK' | 1. SELECT[val:​film:​title:​AIRPORT POLLOCK] <br>2. PROJECT[tbl:​language, #1] <br>3. PROJECT[col:​language:​name, #2] <br> | 1. SELECT[val:​film:​title:​AIRPORT POLLOCK] <br>2. PROJECT[tbl:​language, #1] <br>3. PROJECT[col:​language:​name, #2] <br> | + | medium | 
  | SPIDER_train_2979 | What is the name of the language that the film 'AIRPORT POLLOCK' is in? | SELECT T2.name FROM film AS T1 JOIN LANGUAGE AS T2 ON T1.language_id  =  T2.language_id WHERE T1.title  =  'AIRPORT POLLOCK' | 1. SELECT[tbl:​film] <br>2. PROJECT[tbl:​language, #1] <br>3. COMPARATIVE[#2, #1, comparative:​=:​AIRPORT POLLOCK:​col:​film:​title] <br>4. PROJECT[col:​language:​name, #3] <br> | 1. SELECT[tbl:​film] <br>2. PROJECT[tbl:​language, #1] <br>3. COMPARATIVE[#2, #1, comparative:​=:​AIRPORT POLLOCK:​col:​film:​title] <br>4. PROJECT[col:​language:​name, #3] <br> | + | medium | 
  | SPIDER_train_2980 | How many stores are there? | SELECT count(*) FROM store | 1. SELECT[tbl:​store] <br>2. AGGREGATE[count, #1] <br> | 1. SELECT[tbl:​store] <br>2. AGGREGATE[count, #1] <br> | + | easy | 
  | SPIDER_train_2981 | Count the number of stores. | SELECT count(*) FROM store | 1. SELECT[tbl:​store] <br>2. AGGREGATE[count, #1] <br> | 1. SELECT[tbl:​store] <br>2. AGGREGATE[count, #1] <br> | + | easy | 
  | SPIDER_train_2982 | How many kinds of different ratings are listed? | SELECT count(DISTINCT rating) FROM film | 1. SELECT[col:​film:​rating] <br>2.*(distinct)* PROJECT[different #REF, #1] <br>3. AGGREGATE[count, #2] <br> | 1. SELECT[col:​film:​rating] <br>2.*(distinct)* PROJECT[None, #1] <br>3. AGGREGATE[count, #2] <br> | + | easy | 
  | SPIDER_train_2983 | Count the number of different film ratings. | SELECT count(DISTINCT rating) FROM film | 1. SELECT[tbl:​film] <br>2. PROJECT[col:​film:​rating, #1] <br>3.*(distinct)* FILTER[#2, that are different] <br>4. AGGREGATE[count, #3] <br> | 1. SELECT[tbl:​film] <br>2. PROJECT[col:​film:​rating, #1] <br>3.*(distinct)* COMPARATIVE[#2, #2, None] <br>4. AGGREGATE[count, #3] <br> | + | easy | 
  | SPIDER_train_2986 | How many items in inventory does store 1 have? | SELECT count(*) FROM inventory WHERE store_id  =  1 | 1. SELECT[val:​inventory:​store_id:​1] <br>2. PROJECT[tbl:​inventory, #1] <br>3. PROJECT[tbl:​inventory, #2] <br>4. AGGREGATE[count, #3] <br> | 1. SELECT[val:​rental:​inventory_id:​1] <br>2. PROJECT[tbl:​inventory, #1] <br>3. PROJECT[tbl:​inventory, #2] <br>4. AGGREGATE[count, #3] <br> | - | easy | 
  | SPIDER_train_2987 | Count the number of items store 1 has in stock. | SELECT count(*) FROM inventory WHERE store_id  =  1 | 1. SELECT[col:​inventory:​store_id] <br>2. PROJECT[tbl:​inventory, #1] <br>3. FILTER[#2, comparative:​=:​1:​col:​inventory:​store_id] <br>4. AGGREGATE[count, #3] <br> | 1. SELECT[col:​inventory:​store_id] <br>2. PROJECT[tbl:​inventory, #1] <br>3. COMPARATIVE[#2, #2, comparative:​=:​1:​col:​inventory:​store_id] <br>4. AGGREGATE[count, #3] <br> | + | easy | 
  | SPIDER_train_2991 | Return the address and email of the customer with the first name Linda. | SELECT T2.address ,  T1.email FROM customer AS T1 JOIN address AS T2 ON T2.address_id  =  T1.address_id WHERE T1.first_name  =  'LINDA' | 1. SELECT[tbl:​customer] <br>2. PROJECT[col:​customer:​first_name, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​LINDA:​col:​customer:​first_name] <br>4. PROJECT[col:​address:​address, #3] <br>5. PROJECT[col:​customer:​email, #3] <br>6. UNION[#4, #5] <br> | 1. SELECT[tbl:​customer] <br>2. PROJECT[col:​customer:​first_name, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​LINDA:​col:​customer:​first_name] <br>4. PROJECT[col:​address:​address, #3] <br>5. PROJECT[col:​customer:​email, #3] <br>6. UNION[#4, #5] <br> | + | medium | 
  | SPIDER_train_2994 | What is the first name and the last name of the customer who made the earliest rental? | SELECT T1.first_name ,  T1.last_name FROM customer AS T1 JOIN rental AS T2 ON T1.customer_id  =  T2.customer_id ORDER BY T2.rental_date ASC LIMIT 1 | 1. SELECT[tbl:​customer] <br>2. PROJECT[col:​rental:​rental_date, #1] <br>3. COMPARATIVE[#1, #2, comparative:​min:​None] <br>4. PROJECT[col:​customer:​first_name, #3] <br>5. PROJECT[col:​customer:​last_name, #3] <br>6. UNION[#4, #5] <br> | 1. SELECT[tbl:​customer] <br>2. PROJECT[col:​rental:​rental_date, #1] <br>3. SUPERLATIVE[comparative:​min:​None, #1, #2] <br>4. PROJECT[col:​customer:​first_name, #3] <br>5. PROJECT[col:​customer:​last_name, #3] <br>6. UNION[#4, #5] <br> | + | hard | 
  | SPIDER_train_2997 | Return the full name of the staff who provided a customer with the first name April and the last name Burns with a film rental. | SELECT DISTINCT T1.first_name ,  T1.last_name FROM staff AS T1 JOIN rental AS T2 ON T1.staff_id  =  T2.staff_id JOIN customer AS T3 ON T2.customer_id  =  T3.customer_id WHERE T3.first_name  =  'APRIL' AND T3.last_name  =  'BURNS' | 1. SELECT[tbl:​customer] <br>2. PROJECT[col:​customer:​last_name, #1] <br>3. PROJECT[col:​customer:​last_name, #1] <br>4. COMPARATIVE[#1, #2, comparative:​=:​APRIL:​col:​customer:​first_name] <br>5. COMPARATIVE[#1, #3, comparative:​=:​BURNS:​col:​customer:​last_name] <br>6. INTERSECTION[#1, #4, #5] <br>7. PROJECT[tbl:​rental, #6] <br>8. PROJECT[tbl:​staff, #7] <br>9. PROJECT[col:​staff:​first_name, #8] <br>10. PROJECT[col:​staff:​last_name, #8] <br>11. UNION[#9, #10] <br> | 1. SELECT[tbl:​customer] <br>2. PROJECT[col:​customer:​last_name, #1] <br>3. PROJECT[col:​customer:​last_name, #1] <br>4. COMPARATIVE[#1, #2, comparative:​=:​APRIL:​col:​customer:​first_name] <br>5. COMPARATIVE[#1, #3, comparative:​=:​BURNS:​col:​customer:​last_name] <br>6. INTERSECTION[#1, #4, #5] <br>7. PROJECT[tbl:​rental, #6] <br>8. PROJECT[tbl:​staff, #7] <br>9. PROJECT[col:​staff:​first_name, #8] <br>10. PROJECT[col:​staff:​last_name, #8] <br>11. UNION[#9, #10] <br> | + | hard | 
  | SPIDER_train_2998 | Which store has most the customers? | SELECT store_id FROM customer GROUP BY store_id ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[col:​customer:​store_id] <br>2. PROJECT[tbl:​customer, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | 1. SELECT[col:​customer:​store_id] <br>2. PROJECT[tbl:​customer, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | + | hard | 
  | SPIDER_train_2999 | Return the id of the store with the most customers. | SELECT store_id FROM customer GROUP BY store_id ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[col:​customer:​store_id] <br>2. PROJECT[tbl:​customer, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​customer:​store_id, #4] <br> | 1. SELECT[col:​customer:​store_id] <br>2. PROJECT[tbl:​customer, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​customer:​store_id, #4] <br> | + | hard | 
  | SPIDER_train_3001 | Return the amount of the largest payment. | SELECT amount FROM payment ORDER BY amount DESC LIMIT 1 | 1. SELECT[tbl:​payment] <br>2. PROJECT[col:​payment:​amount, #1] <br>3. SUPERLATIVE[comparative:​max:​None, #1, #2] <br>4. PROJECT[col:​payment:​amount, #3] <br> | 1. SELECT[tbl:​payment] <br>2. PROJECT[col:​payment:​amount, #1] <br>3. SUPERLATIVE[comparative:​max:​None, #1, #2] <br>4. PROJECT[col:​payment:​amount, #3] <br> | + | medium | 
 ***
 Exec acc: **0.9048**
