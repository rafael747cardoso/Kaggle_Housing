
### Check if the vector is a dummy

is_dummy = function(x){
    unique_x = unique(x)
    unique_x = unique_x[!is.na(unique_x)]
    if(unique_x[1] == 0 &
       unique_x[2] == 1 &
       length(unique_x) == 2){
        return(TRUE)
    } else{
        return(FALSE)
    }
}

