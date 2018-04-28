<?php
  include "header.php";
?> 
<link href="css/color-picker.min.css" rel="stylesheet">
<script src="js/color-picker.min.js"></script> 

<div class="wallheader"> 
   <div class="container">
       <h1 style="margin-top:0px; padding-top:10px;padding-bottom:10px;"><a href="index.php">Emotion Recognition</a></h1>
   </div>
</div>

<?php
    function fn_safe($in) {
        $in = trim($in);
        $in = stripslashes($in); 
        $in = htmlspecialchars($in); 
        return $in;
      }
  $imgpath = fn_safe($_GET['img']); 

  $newimagepath = "./uploads/". $imgpath; 
  $command = '/home/chuhaoyu/anaconda3/bin/python3.6 ./emotion_recognition.py --path='.$newimagepath; 
  $result = shell_exec($command.' 2>&1'); 
  $result = str_replace("\n","<br>",$result);
?>

<div class="examples container">
    <h2 style="width: 650px;float: left"> Input</h2> <h2 style="width: 400px;float:left"> Results</h2>
   <div class="row">
     <div class="col-xs-6 col-md-4"> <img src="<?php echo $newimagepath;?>"/> </div>
     <div class="col-xs-6 col-md-4" style="margin-left: 260px; color: #FF3300"> <?php echo $result;?> </div>
   </div>

</div> 

<?php 
  include "footer.php"
?>
