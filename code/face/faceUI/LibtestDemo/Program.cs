using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;


namespace LibtestDemo
{
    static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();


            // string img = "src2.jpg"; //目标图像路径

            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());
        }
        
    }
}
