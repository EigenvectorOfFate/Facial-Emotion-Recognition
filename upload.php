<?php
function renamemd5($filename)
{
   $ext = explode('.',$filename);
   return md5(uniqid(microtime())).'.'.$ext[1];
}
function uploadimage($target_dir, $type_name)
{ 
	if(!empty($_FILES))
        {

                $name = renamemd5(basename($_FILES[$type_name]["name"]));
		$target_file = $target_dir.$name; 

		$imageFileType = pathinfo($target_file,PATHINFO_EXTENSION);
	       
		// Check if image file is a actual image or fake image
		if(isset($_POST["submit"])) {
		    $check = getimagesize($_FILES[$type_name]["tmp_name"]);
		    if($check == false) { 
			echo "<script> alert('File is not an image.'); history.back(); </script>";
			 return null;
		    }
		} 
		// Check file size
		if ($_FILES[$type_name]["size"] > 1024*1024*5) { 
		    echo "<script> alert('Sorry, your file is too large. ( No more than 5M .)'); history.back(); </script>";
		     return null;
		}
		// Allow certain file formats
		if(!strcasecmp($imageFileType,"jpg") && !strcasecmp($imageFileType,"png") && !strcasecmp($imageFileType,"jpeg")
		&& !strcasecmp($imageFileType,"gif") ) { 
		    echo "<script> alert('Sorry, only JPG, JPEG, PNG & GIF files are allowed.'); history.back(); </script>";
		    return null;
		}
		// Check if $uploadOk is set to 0 by an error
		 
		    if (! move_uploaded_file($_FILES[$type_name]["tmp_name"], $target_file)) {  
		        echo "<script> alert('Sorry, there was an error uploading your file.'); history.back(); </script>";
		    }else{
			return $name;
		    } 
	}
}
?>
