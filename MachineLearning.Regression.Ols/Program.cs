using System;
using System.Linq;
using Microsoft.ML;

namespace MachineLearning.Regression.Ols
{
    public static class Program
    {
        public static void Main()
        {
            ExampleOfLinearModelOf1Variable();
            ExampleOfLinearModelOf3Variable();
            ExampleOfLinearModelOf1VariableWithDisturbance();
            Console.ReadKey();
        }

        private static void ExampleOfLinearModelOf1Variable()
        {
            var observations =
                Enumerable.Range(1, 5)
                    .Select(x => new LinearFunctionOf1Variable(x));
            var mlContext =
                new MLContext();
            var data =
                mlContext.Data.LoadFromEnumerable(observations);
            var pipeline =
                mlContext.Transforms
                    .Concatenate(
                        "Features",
                        nameof(LinearFunctionOf1Variable.X))
                    .Append(
                        mlContext.Regression.Trainers.Ols(
                            labelColumnName: nameof(LinearFunctionOf1Variable.Y)));
            var model =
                pipeline.Fit(data);
            var metrics =
                mlContext.Regression.Evaluate(
                    data,
                    labelColumnName: nameof(LinearFunctionOf1Variable.Y),
                    scoreColumnName: nameof(Prediction.Y));
            new RegressionResults(
                    "Result for linear function of 1 variable",
                    model.LastTransformer.Model,
                    metrics)
                .PrintToConsole();
        }

        private static void ExampleOfLinearModelOf3Variable()
        {
            var observations =
                new[]
                {
                    new LinearFunctionOf3Variable(1, 1, 1),
                    new LinearFunctionOf3Variable(1, 1, 2),
                    new LinearFunctionOf3Variable(1, 2, 2),
                    new LinearFunctionOf3Variable(2, 2, 2),
                    new LinearFunctionOf3Variable(2, 2, 1),
                    new LinearFunctionOf3Variable(2, 1, 1),
                };
            var mlContext =
                new MLContext();
            var data =
                mlContext.Data.LoadFromEnumerable(observations);
            var pipeline =
                mlContext.Transforms
                    .Concatenate(
                        "Features",
                        nameof(LinearFunctionOf3Variable.X1),
                        nameof(LinearFunctionOf3Variable.X2),
                        nameof(LinearFunctionOf3Variable.X3))
                    .Append(
                        mlContext.Regression.Trainers.Ols(
                            labelColumnName: nameof(LinearFunctionOf3Variable.Y)));
            var model =
                pipeline.Fit(data);
            var metrics =
                mlContext.Regression.Evaluate(
                    data,
                    labelColumnName: nameof(LinearFunctionOf3Variable.Y),
                    scoreColumnName: nameof(Prediction.Y));
            new RegressionResults(
                    "Result for linear function of 3 variable",
                    model.LastTransformer.Model,
                    metrics)
                .PrintToConsole();
        }

        private static void ExampleOfLinearModelOf1VariableWithDisturbance()
        {
            var random =
                new Random(Environment.TickCount);
            // Function generates disturbance from range <-0.05, 0.05>
            Func<float> disturbance =
                () => random.Next(-50, 51) / (float)1000;
            var observations =
                Enumerable.Range(1, 5)
                    .Select(x => new LinearFunctionOf1VariableWithDisturbance(x, disturbance()));
            var mlContext =
                new MLContext();
            var data =
                mlContext.Data.LoadFromEnumerable(observations);
            var pipeline =
                mlContext.Transforms
                    .Concatenate(
                        "Features",
                        nameof(LinearFunctionOf1Variable.X))
                    .Append(
                        mlContext.Regression.Trainers.Ols(
                            labelColumnName: nameof(LinearFunctionOf1Variable.Y)));
            var model =
                pipeline.Fit(data);
            var metrics =
                mlContext.Regression.Evaluate(
                    data,
                    labelColumnName: nameof(LinearFunctionOf1Variable.Y),
                    scoreColumnName: nameof(Prediction.Y));
            new RegressionResults(
                    "Result for linear function of 1 variable with disturbance",
                    model.LastTransformer.Model,
                    metrics)
                .PrintToConsole();
        }
    }
}
