Regarding the new approach, below is the high level thought, please do let me know for more discussions required.
1) All address fields will be concatenated and will be considered as one wholesome address entity
2) Fields will be extracted from this address for city, state ,country and pincode
3) There will be rules running for each of these fields and will score the confidence for each rule
4) The rules will be 
a. to check either an exact or fuzzy match on the values, 
b. to match values as a combination, for eg) city, state and country against the identified pincode
c. to tag ambiguities wherever unique state, city , country or pincode may not be derived.
