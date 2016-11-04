library(animation)
library(plotrix)

# Load the data. Source: US Census Bureau
tag_relevence <- read.csv("data/Tag_July_hisCall.csv", stringsAsFactors=FALSE)

# Temporary image directory
ani.options(outdir = paste(getwd(), "/images", sep=""))

saveGIF( 
for (day in unique(tag_relevence$date)) {
  tag_day <- subset(tag_relevence, date==day)
  day_call_1 <- tag_day$call_1
  day_call_2 <- tag_day$call_2

  par(cex=1.0)
  par(
      mar=pyramid.plot(day_call_1, day_call_2, 
      top.labels=c("Trump", "", "Hillary"), 
      labels=tag_day$tag, main=day, lxcol="#A097CC", rxcol="#EDBFBE", 
      xlim=c(60,60), 
      gap=20,unit="number of total call cumulated / thousands")
     )
}, 
  movie.name = "tag_july_HisCall.gif", interval=2, 
  nmax=50, ani.width=800, ani.height=400)



