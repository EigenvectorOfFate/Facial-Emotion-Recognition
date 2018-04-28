<?php 
  include "upload.php"; 

    function fn_safe($in) {
        $in = trim($in);
        $in = stripslashes($in); 
        $in = htmlspecialchars($in); 
        return $in;
      }
    $application = fn_safe($_GET['a']); 
   
   if($application == 'file')
   {
	  $target_dir = "./uploads/";
	  $imagename = uploadimage($target_dir, 'fileToUpload');
	  $imagepath = $target_dir.$imagename;

	 if($imagename != null)
	 { 
      #$command = '/home/chuhaoyu/anaconda3/bin/python3.6 ./emotion_recognition.py --path='.$imagepath; 
      #$result = shell_exec($command.' 2>&1'); 
      #$result = str_replace("\n","<br>",$result);
      echo( " <script language='javascript' type='text/javascript'> window.location.href='returnlabel.php?img=".$imagename."'; </script>");
      #echo $result;
	 }else{
	    echo "<script> alert('Sorry, there was an error.'); history.back(); </script>";
	 } 
     }else if($application == 'test'){
       
        $testimage = $_POST["testSelect"];

        $target_dir = "./img/test/";
        if($testimage == "1"){
            $imagename = '1.png';
         }else if($testimage == "2"){
            $imagename = '2.png';
         }else if($testimage == "3"){
            $imagename = '3.png';
         }else if($testimage == "4"){
            $imagename = '4.png';
         }else if($testimage == "5"){
            $imagename = '5.png';
         }else if($testimage == "6"){
            $imagename = '6.png';
         }else if($testimage == "7"){
            $imagename = '7.png';
         }else{
            echo "<script> alert('Sorry, there are some errors.'); history.back(); </script>";
         } 
        $imagepath = $target_dir.$imagename;


        echo( " <script language='javascript' type='text/javascript'> window.location.href='returnlabel.php?img=".$imagename."'; </script>");

     }else{
           echo "<script> history.back(); </script>";
      }
?>


