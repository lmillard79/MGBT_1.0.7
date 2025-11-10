/*
 * C# MGBT Wrapper for Python Integration
 * 
 * This is a standalone C# program that uses the USACE-RMC Numerics library
 * to run MGBT and output results in a format Python can parse.
 * 
 * Usage: csharp_mgbt_wrapper.exe <flow1> <flow2> <flow3> ...
 * Output: JSON with klow and threshold
 * 
 * To use this:
 * 1. Clone USACE-RMC/Numerics from GitHub
 * 2. Build the Numerics library
 * 3. Compile this wrapper referencing Numerics.dll
 * 4. Place the executable in the scripts directory
 * 
 * Compilation example:
 * csc /reference:Numerics.dll /out:csharp_mgbt_wrapper.exe csharp_mgbt_wrapper.cs
 */

using System;
using System.Linq;
using Numerics.Data.Statistics;

namespace MGBTWrapper
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                if (args.Length == 0)
                {
                    Console.Error.WriteLine("Error: No flow values provided");
                    Console.WriteLine("{\"error\": \"No flow values provided\"}");
                    Environment.Exit(1);
                }

                // Parse flow values
                double[] flows = args.Select(double.Parse).ToArray();

                // Run MGBT using USACE-RMC Numerics implementation
                int klow = MultipleGrubbsBeckTest.Function(flows);

                // Calculate threshold (lowest non-outlier value)
                double threshold = double.NaN;
                if (klow > 0 && klow < flows.Length)
                {
                    var sortedFlows = flows.OrderBy(x => x).ToArray();
                    threshold = sortedFlows[klow];
                }

                // Output as JSON for Python to parse
                Console.WriteLine($"{{\"klow\": {klow}, \"threshold\": {threshold}}}");
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error: {ex.Message}");
                Console.WriteLine($"{{\"error\": \"{ex.Message}\"}}");
                Environment.Exit(1);}
        }
    }
}
