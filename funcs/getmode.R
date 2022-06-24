
### Get the mode of a vector

getmode = function(v){
   unique_v = unique(v)
   unique_v = unique_v[!is.na(unique_v)]
   unique_v[which.max(tabulate(match(v, unique_v)))]
}
