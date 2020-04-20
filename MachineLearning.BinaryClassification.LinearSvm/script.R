data = read.csv("iris.data", header = FALSE)
colnames(data) = c("sepal_length", "sepal_width", "petal_length", "petal_width", "class")

# Print summary of data frame.
summary(data)

pairs(
  ~ sepal_length + sepal_width + petal_length + petal_width,
  data = data,
  col = data$class)

# Generate plot
plot(
  data$petal_length,
  data$sepal_length,
  col = data$class,
  xlab = "petal length",
  ylab = "sepal length")
legend(
  "topright",
  legend = unique(data$class),
  col = 1:length(data$class),
  pch= 1)

