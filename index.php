<!DOCTYPE html>

<html lang="en">
<?php include 'header.php'; ?>  
<body>  
<div class="uploadBox">
&nbsp
    <h1>Emotion Recognition</h1> 
    <div class="introductionbox">
      <p class="lead">This work focuses on emotion recognition using the all convolutional net. Given an image of the person, it can return the emotion label. You can click "SELECT A FILE" button to choose a picture that you want to know which emtion is appear on his/her face or choose one of the 7 test pictures below to show you the performance of the system.The accuracy of the system on Toronto Face Dataset(TFD) test set is 86.7%.</p>
    </div>
    <div class="container"> 
        <div class="col-md-8 col-md-offset-3" style="padding-bottom:10px">
	<form method="post" action="returnweb.php?a=file" id="upload" enctype="multipart/form-data">    
	   <div class="custom-file-upload"> 
	     <input type="file" id="fileToUpload" name="fileToUpload" />
             <button type="submit" id="filesubmit" class="filesubmit"> Start </button>
	   </div>  
	</form>
        </div> 
    </div>

</div>

<div class="container">
     <div class="row">
	 <div class="picker">
             <form method="post" action="returnweb.php?a=test" id="upload" enctype="multipart/form-data"> 
              <label style="font-size:24px"> Select an image for test: 
</label> 
	      <select class="image-picker show-labels show-html" id="testSelect" name="testSelect">
		<option data-img-label="Happy" data-img-src="img/examples/happy.png" value="1">&nbsp Test image 1 &nbsp </option>
		<option data-img-label="Surprise" data-img-src="img/examples/surprise.png" value="2">&nbsp Test image 2 &nbsp </option>
		<option data-img-label="Neutral" data-img-src="img/examples/neutral.png" value="3">&nbsp Test image 3 &nbsp </option>
		<option data-img-label="Fear" data-img-src="img/examples/fear.png" value="4">&nbsp Test image 4 &nbsp </option>
    <option data-img-label="Sad" data-img-src="img/examples/sad.png" value="5">&nbsp Test image 5 &nbsp </option>
		<option data-img-label="Disgust" data-img-src="img/examples/disgust.png" value="6">&nbsp Test image 6 &nbsp </option>
		<option data-img-label="Annoyed" data-img-src="img/examples/annoyed.png" value="7">&nbsp Test image 7 &nbsp </option>
	      </select>  
              <button type="submit" class="btn btn-success" style="font-size:24px">&nbsp START TEST &nbsp</button>  
             </form>
	</div> 
    </div>

<script src="https://code.jquery.com/jquery-1.12.4.min.js"></script>
<script src="js/image-picker.js" type="text/javascript"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script> 

<script>
	(function($) { 
		  // Browser supports HTML5 multiple file?
		  var multipleSupport = typeof $('<input/>')[0].multiple !== 'undefined',
		      isIE = /msie/i.test( navigator.userAgent );

		  $.fn.customFile = function() {

		    return this.each(function() {

		      var $file = $(this).addClass('custom-file-upload-hidden'), // the original file input
			  $wrap = $('<div class="file-upload-wrapper">'),
			  $input = $('<input type="text" class="file-upload-input" />'),
			  // Button that will be used in non-IE browsers
			  $button = $('<button type="button" class="file-upload-button">Select a File</button>'),
			  // Hack for IE
			  $label = $('<label class="file-upload-button" for="'+ $file[0].id +'">Select a File</label>');

		      // Hide by shifting to the left so we
		      // can still trigger events
		      $file.css({
			position: 'absolute',
			left: '-9999px'
		      });

		      $wrap.insertAfter( $file )
			.append( $file, $input, ( isIE ? $label : $button ) );

		      // Prevent focus
		      $file.attr('tabIndex', -1);
		      $button.attr('tabIndex', -1);

		      $button.click(function () {
			$file.focus().click(); // Open dialog
		      });

		      $file.change(function() {

			var files = [], fileArr, filename;

			// If multiple is supported then extract
			// all filenames from the file array
			if ( multipleSupport ) {
			  fileArr = $file[0].files;
			  for ( var i = 0, len = fileArr.length; i < len; i++ ) {
			    files.push( fileArr[i].name );
			  }
			  filename = files.join(', ');

			// If not supported then just take the value
			// and remove the path to just show the filename
			} else {
			  filename = $file.val().split('\\').pop();
			}

			$input.val( filename ) // Set the value
			  .attr('title', filename) // Show filename in title tootlip
			  .focus(); // Regain focus

		      });

		      $input.on({
			blur: function() { $file.trigger('blur'); },
			keydown: function( e ) {
			  if ( e.which === 13 ) { // Enter
			    if ( !isIE ) { $file.trigger('click'); }
			  } else if ( e.which === 8 || e.which === 46 ) { // Backspace & Del
			    // On some browsers the value is read-only
			    // with this trick we remove the old input and add
			    // a clean clone with all the original events attached
			    $file.replaceWith( $file = $file.clone( true ) );
			    $file.trigger('change');
			    $input.val('');
			  } else if ( e.which === 9 ){ // TAB
			    return;
			  } else { // All other keys
			    return false;
			  }
			}
		      });

		    });

		  };

		  // Old browser fallback
		  if ( !multipleSupport ) {
		    $( document ).on('change', 'input.customfile', function() {

		      var $this = $(this),
			  // Create a unique ID so we
			  // can attach the label to the input
			  uniqId = 'customfile_'+ (new Date()).getTime(),
			  $wrap = $this.parent(),

			  // Filter empty input
			  $inputs = $wrap.siblings().find('.file-upload-input')
			    .filter(function(){ return !this.value }),

			  $file = $('<input type="file" id="'+ uniqId +'" name="'+ $this.attr('name') +'"/>');

		      // 1ms timeout so it runs after all other events
		      // that modify the value have triggered
		      setTimeout(function() {
			// Add a new input
			if ( $this.val() ) {
			  // Check for empty fields to prevent
			  // creating new inputs when changing files
			  if ( !$inputs.length ) {
			    $wrap.after( $file );
			    $file.customFile();
			  }
			// Remove and reorganize inputs
			} else {
			  $inputs.parent().remove();
			  // Move the input so it's always last on the list
			  $wrap.appendTo( $wrap.parent() );
			  $wrap.find('input').focus();
			}
		      }, 1);

		    });
		  }

}(jQuery)); 
$('input[type=file]').customFile();
</script> 
<script type="text/javascript">

    jQuery("select.image-picker").imagepicker({
      hide_select:  false,
    });

    jQuery("select.image-picker.show-labels").imagepicker({
      hide_select:  false,
      show_label:   true,
    });

    jQuery("select.image-picker.limit_callback").imagepicker({
      limit_reached:  function(){alert('We are full!')},
      hide_select:    false
    });

    var container = jQuery("select.image-picker.masonry").next("ul.thumbnails");
    container.imagesLoaded(function(){
      container.masonry({
        itemSelector:   "li",
      });
    });

</script> 
</body>
<?php include 'footer.php'; ?>  
</html>
