
### Fix the levels

fix_bad_levels = function(x){
    x = as.factor(x)
    lvls = levels(x)
    lvls = gsub(x = lvls,
                pattern = "/",
                replacement = "_")
    lvls = gsub(x = lvls,
                pattern = " ",
                replacement = "_")
    lvls = gsub(x = lvls,
                pattern = "-",
                replacement = "_")
    lvls = gsub(x = lvls,
                pattern = "\\(",
                replacement = "")
    lvls = gsub(x = lvls,
                pattern = "\\)",
                replacement = "")
    lvls = gsub(x = lvls,
                pattern = "\\.",
                replacement = "_")
    lvls = gsub(x = lvls,
                pattern = "\\&",
                replacement = "_")
    levels(x) = lvls
    x = as.character(x)
    return(x)
}
