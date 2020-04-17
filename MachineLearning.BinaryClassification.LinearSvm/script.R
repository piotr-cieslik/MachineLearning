data = read.csv("iris.data", header = FALSE)
colnames(data) = c("sepal_length", "sepal_width", "petal_length", "petal_width", "class")

# Print summary of data frame.
summary(data)

# Initialize grapihc device.
png("sepal_length_to_sepal_width.png")

# Generate plot
plot(
  data$sepal_length,
  data$sepal_width,
  col = data$class,
  xlab = "sepal length",
  ylab = "sepal width")
legend(
  "topright",
  legend = unique(data$class),
  col = 1:length(data$class),
  pch= 1)

# Shut down the device and save file.
dev.off()
