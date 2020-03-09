using Microsoft.ML.Data;

namespace MachineLearning.Regression.FastTree
{
    public sealed class WineData
    {
        [LoadColumn(0)]
        public float Label { get; set; }

        [LoadColumn(1)]
        public float Alcohol { get; set; }

        [LoadColumn(2)]
        public float MalicAcid { get; set; }

        [LoadColumn(3)]
        public float Ash { get; set; }

        [LoadColumn(4)]
        public float AlcalinityOfAash { get; set; }

        [LoadColumn(5)]
        public float Magnesium { get; set; }

        [LoadColumn(6)]
        public float TotalPhenols { get; set; }

        [LoadColumn(7)]
        public float Flavanoids { get; set; }

        [LoadColumn(8)]
        public float NonflavanoidPhenols { get; set; }

        [LoadColumn(9)]
        public float Proanthocyanins { get; set; }

        [LoadColumn(10)]
        public float ColorIntensity { get; set; }

        [LoadColumn(11)]
        public float Hue { get; set; }

        [LoadColumn(12)]
        public float OD280_OD315_OfDilutedWines { get; set; }

        [LoadColumn(13)]
        public float Proline { get; set; }
    }
}
