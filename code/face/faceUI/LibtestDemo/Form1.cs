using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Xml.Linq;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Diagnostics;

//this.pictureBox1.Paint += PictureBox1_Paint;


namespace LibtestDemo
{
    public partial class Form1 : Form
    {


        [DllImport("face.dll", EntryPoint = "FaceID", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern int FaceID(StringBuilder Imgpath, StringBuilder dirpath);

        [DllImport("face.dll", EntryPoint = "init", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern void init();

        [DllImport("face.dll", EntryPoint = "FaceID1", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern int FaceID1(StringBuilder Imgpath, IntPtr p, int rows, int cols);
      

        public Form1()
        {
            InitializeComponent();
            
        }
        Image imgshow;// = Image.FromFile("s.jpg");
        bool f = false;
        bool f1 = true;
        private void PictureBox1_Paint(object sender, PaintEventArgs e)
        {
           if (f)
            {

                Invalidate();
                e.Graphics.DrawImage(imgshow,0,0);

            }

        }


        private void button1_Click(object sender, EventArgs e)
        {

            // Image imgshow0 = Image.FromFile("src.jpg");
            //  pictureBox1.Image = imgshow0;
            //   StringBuilder imagepth = new StringBuilder("src2.jpg");
            //StringBuilder dirpth = new StringBuilder("0.jpg");
            //int cnt = FaceID(imagepth, dirpth);
           
            Thread threadA = new Thread(run_cap);
            threadA.Start();

           // int cnt = FaceID1(imagepth, src.Data, src.Rows, src.Cols);
         

        }

        void run_cap()
        {
            init();
            int k = 0;
            Mat src = new Mat();
            FrameSource frame = Cv2.CreateFrameSource_Camera(0);
            StringBuilder imagepth = new StringBuilder("src2.jpg");
            while (f1)
            {

                for (int i = 0; i <= 100000; i++) ;

                frame.NextFrame(src);
                int cnt = FaceID1(imagepth, src.Data, src.Rows, src.Cols);
                Bitmap bitmap = BitmapConverter.ToBitmap(src);
                //Cv2.ImWrite("src.jpg", src);
                //StringBuilder imagepth = new StringBuilder("src2.jpg");
                //StringBuilder dirpth = new StringBuilder("src.jpg");
                //int cnt = FaceID(imagepth, dirpth);
                //textBox1.Text = "out:\n" + k;k++;
                // textBox1.AppendText("add:\n");
                if (cnt == -1) { textBox1.AppendText("AI: No one! \n"); }
                if (cnt == 0) { textBox1.AppendText("AI: Who are you ? \n"); }
                if (cnt == 1) { textBox1.AppendText("AI: Hello, master! \n"); }
                Invalidate();
                pictureBox1.Invalidate();
                imgshow = bitmap;
                f = true;
            }

        }



        private void button2_Click(object sender, EventArgs e)
        {
            f1 = false;

        }

  

        private void button3_Click(object sender, EventArgs e)
        {
      

            Bitmap bitm1 = new Bitmap(imgshow);
            System.Drawing.Graphics g = System.Drawing.Graphics.FromImage(bitm1);
            
            bitm1.Save("s.jpg", System.Drawing.Imaging.ImageFormat.Jpeg);

 
        }

        //private void textBox1_TextChanged(object sender, KeyPressEventArgs e)
        //{
        //    if (e.KeyChar == '\r')
        //    {
        //        int num = textBox1.GetFirstCharIndexOfCurrentLine();
        //        string current = textBox1.Lines[num];
        //        textBox1.AppendText("\n got it");
                
        //        e.Handled = true;
        //    }
        //}

        private void textBox2_TextChanged(object sender, EventArgs e)
        {

            //int num = textBox1.GetFirstCharIndexOfCurrentLine();
            //string current = textBox1.Lines[num];
            //textBox2.Clear();
            string current = textBox2.Text;

            if (current.EndsWith("\n"))
            {
                textBox1.AppendText("Me:" + current + "\n");
                textBox2.Clear();
            }


        }
    }
}








