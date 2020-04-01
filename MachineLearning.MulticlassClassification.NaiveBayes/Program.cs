using Microsoft.ML;
using System;

namespace MachineLearning.MulticlassClassification.NaiveBayes
{
    class Program
    {
        static void Main()
        {
            var mlContext =
                new MLContext(seed: 0);

            var irisData =
                mlContext.Data.LoadFromTextFile<CarData>(
                    "car.data",
                    separatorChar: ',');
            var data =
                mlContext.Data.TrainTestSplit(
                    irisData,
                    testFraction: 0.2);

            // Important note.
            // It's hard to believe but according to documentation NaiveBayer classifier
            // requires vector of binary features of type float, where:
            // - x <= 0 are false,
            // - x > 0 are true.
            // https://docs.microsoft.com/pl-pl/dotnet/api/microsoft.ml.standardtrainerscatalog.naivebayes
            var pipeline =
                    mlContext.Transforms.CustomMapping<CarData, CarDataTransformed>(
                    (x, y) =>
                    {
                        y.Label = x.Class;

                        y.BuyingVHigh = x.Buying == "vhigh" ? 1 : -1;
                        y.BuyingHigh = x.Buying == "high" ? 1 : -1;
                        y.BuyingMed = x.Buying == "med" ? 1 : -1;
                        y.BuyingLow = x.Buying == "low" ? 1 : -1;

                        y.MaintVHigh = x.Maint == "vhigh" ? 1 : -1;
                        y.MaintHigh = x.Maint == "high" ? 1 : -1;
                        y.MaintMed = x.Maint == "med" ? 1 : -1;
                        y.MaintLow = x.Maint == "low" ? 1 : -1;

                        y.Doors2 = x.Doors == "2" ? 1 : -1;
                        y.Doors3 = x.Doors == "3" ? 1 : -1;
                        y.Doors4 = x.Doors == "4" ? 1 : -1;
                        y.Doors5AndMore = x.Doors == "5more" ? 1 : -1;

                        y.Persons2 = x.Persons == "2" ? 1 : -1;
                        y.Persons4 = x.Persons == "4" ? 1 : -1;
                        y.PersonsMore = x.Persons == "more" ? 1 : -1;

                        y.LugBootSmall = x.LugBoot == "small" ? 1 : -1;
                        y.LugBootMed = x.LugBoot == "med" ? 1 : -1;
                        y.LugBootBig = x.LugBoot == "big" ? 1 : -1;

                        y.SafetyLow = x.Safety == "low" ? 1 : -1;
                        y.SafetyMed = x.Safety == "med" ? 1 : -1;
                        y.SafetyHigh = x.Safety == "high" ? 1 : -1;
                    },
                    contractName: default)
                .Append(
                    mlContext.Transforms.Concatenate(
                        "Features",
                        nameof(CarDataTransformed.BuyingVHigh),
                        nameof(CarDataTransformed.BuyingHigh),
                        nameof(CarDataTransformed.BuyingMed),
                        nameof(CarDataTransformed.BuyingLow),
                        nameof(CarDataTransformed.MaintVHigh),
                        nameof(CarDataTransformed.MaintHigh),
                        nameof(CarDataTransformed.MaintMed),
                        nameof(CarDataTransformed.MaintLow),
                        nameof(CarDataTransformed.Doors2),
                        nameof(CarDataTransformed.Doors3),
                        nameof(CarDataTransformed.Doors4),
                        nameof(CarDataTransformed.Doors5AndMore),
                        nameof(CarDataTransformed.Persons2),
                        nameof(CarDataTransformed.Persons4),
                        nameof(CarDataTransformed.PersonsMore),
                        nameof(CarDataTransformed.LugBootSmall),
                        nameof(CarDataTransformed.LugBootMed),
                        nameof(CarDataTransformed.LugBootBig),
                        nameof(CarDataTransformed.SafetyLow),
                        nameof(CarDataTransformed.SafetyMed),
                        nameof(CarDataTransformed.SafetyHigh)))
                .Append(
                    mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .Append(
                    mlContext.MulticlassClassification.Trainers.NaiveBayes());

            var model =
                pipeline.Fit(data.TrainSet);

            var prediction =
                model.Transform(data.TestSet);

            var metrics =
                mlContext.MulticlassClassification.Evaluate(prediction);
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }
    }
}
