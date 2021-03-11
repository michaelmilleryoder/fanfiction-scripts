library('stm')

# Settings
num_topics <- 20
max_iters <- 50
do_stem <- TRUE
min_df <- 100 # minimum document frequency for words
covariates <- 'days'

cat('Loading data...\n')
data <- read.csv('/data/news/now2010-2021/lgbtq/articles.csv')
data$country <- as.factor(data$country)
data$source <- as.factor(data$source)
data$date <- as.Date(data$date)
data$day <- julian(data$date, origin=as.Date('2010-01-01'))

cat('Processing data...\n')
processed <- textProcessor(data$text, stem=do_stem, metadata=data,
		verbose=TRUE)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta,
					lower.thresh=min_df)

cat('\nEstimating STM...\n')
estimated <- stm(documents = out$documents,
                vocab = out$vocab,
                K = num_topics,
                max.em.its = max_iters,
                prevalence =~ day,
				verbose=TRUE,
                data = out$meta)

cat('Saving model...\n')
outpath <- sprintf('/projects/fanfiction_lgbtq/models/lgbtq_news_stm_%s_%dtopics_%dit_%dmindf.rds', covariates, num_topics, max_iters, min_df)
saveRDS(estimated, outpath)
