'use strict';
var pathname = window.location.pathname; // Returns path only
var link_en = $('#en');
if(pathname.endsWith("en.html")){
    var pathname_cn = pathname.replace("_en.html",".html");
    console.log(pathname_cn);
    $('cn').attr("href",pathname_cn);
}else{

}
