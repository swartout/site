---
title: My Website!
---

# How my website and blog works!


I've always a little piece of the internet that's mine - and if I'm going to
write more here, I'd love for it to be on something I created. Until recently,
my website has been a Jekyll template hosted on GitHub Pages. The [old
blog](https://blog.cswartout.com) has worked well, but I still wanted something
I could customize more and feel more personal.

The source code is at [this repo](https://github.com/swartout/site) and the
mentioned scripts are linked.

## What I want with the new website

*Point 1.* The Jekyll template allowed me to make posts using markdown, something I
enjoyed. It's quite easy to write and has the advantage of being able to support
fancy-looking LaTeX commands I use for my notes: $\pi^2 = 10$

*Point 2.* The site should be able to host my old posts as well as my class notes ([old
location](https://blog.cswartout.com/notes)), which are also written in
markdown. (Aside: if you type out your class notes, I highly recommend posting
them online. It's always easy to access and others can see it, which is great
motivation!) The GitHub Pages site automatically updated whenever I pushed a
commit - this should be equally easy!

*Point 3.* It should be simple enough for me to understand and quickly try out
changes. I don't want to spend 30 minutes trying to get why some template is
configured the way it is.

Finally, it should feel like something I made!

## My solution

I discovered [Pandoc](https://pandoc.org/) while taking a class and wanted to
use it for this. I took one of my [old blog
posts](https://github.com/swartout/site/blob/main/posts/gpt1.md?plain=1) and
ran it through pandoc to convert it to html, and it worked pretty well!
I made some modifications to the default template used, mainly including a
header, supporting LaTeX, and adding some CSS. Now, I could write pages in
markdown and quickly convert it to html!

To create the all of the html files on each update, I wrote a simple bash script
([`build.sh`](https://raw.githubusercontent.com/swartout/site/main/build.sh)) to
iterate and convert all files in specified directories, and voila! I've got a
working build process!

To host this site, I've got a AWS Lightsail server which costs ~$5/month. I
have another simple script
([`update.sh`](https://raw.githubusercontent.com/swartout/site/main/update.sh))
which pulls all updates from GitHub, then builds the site. Whenever I want to
add something, I just need to push it to GitHub, then update the site on the
server via ssh: `ssh lightsail-server /var/www/site/update.sh`

I'm looking forward to writing more (substantial) posts on here :) Carter
