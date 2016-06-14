

/*-------------------------------------------------------------------------*/
/**
  @file		gnuplot_i.c
  @author	N. Devillard
  @date	Sep 1998
  @version	$Revision: 2.10 $
  @brief	C interface to gnuplot.

  gnuplot is a freely available, command-driven graphical display tool for
  Unix. It compiles and works quite well on a number of Unix flavours as
  well as other operating systems. The following module enables sending
  display requests to gnuplot through simple C calls.
  
*/
/*--------------------------------------------------------------------------*/

/*
	$Id: gnuplot_i.c,v 2.10 2003/01/27 08:58:04 ndevilla Exp $
	$Author: ndevilla $
	$Date: 2003/01/27 08:58:04 $
	$Revision: 2.10 $
 */

/*---------------------------------------------------------------------------
                                Includes
 ---------------------------------------------------------------------------*/

#include "gnuplot_i.h"
#include <math.h>

/*---------------------------------------------------------------------------
                                Defines
 ---------------------------------------------------------------------------*/

/** Maximal size of a gnuplot command */
#define GP_CMD_SIZE     	20048
/** Maximal size of a plot title */
#define GP_TITLE_SIZE   	80
/** Maximal size for an equation */
#define GP_EQ_SIZE      	512
/** Maximal size of a name in the PATH */
#define PATH_MAXNAMESZ       4096

/** Define P_tmpdir if not defined (this is normally a POSIX symbol) */
#ifndef P_tmpdir
#define P_tmpdir "."
#endif

/*---------------------------------------------------------------------------
                            Function codes
 ---------------------------------------------------------------------------*/


/*-------------------------------------------------------------------------*/
/**
  @brief	Find out where a command lives in your PATH.
  @param	pname Name of the program to look for.
  @return	pointer to statically allocated character string.

  This is the C equivalent to the 'which' command in Unix. It parses
  out your PATH environment variable to find out where a command
  lives. The returned character string is statically allocated within
  this function, i.e. there is no need to free it. Beware that the
  contents of this string will change from one call to the next,
  though (as all static variables in a function).

  The input character string must be the name of a command without
  prefixing path of any kind, i.e. only the command name. The returned
  string is the path in which a command matching the same name was
  found.

  Examples (assuming there is a prog named 'hello' in the cwd):

  @verbatim
  gnuplot_get_program_path("hello") returns "."
  gnuplot_get_program_path("ls") returns "/bin"
  gnuplot_get_program_path("csh") returns "/usr/bin"
  gnuplot_get_program_path("/bin/ls") returns NULL
  @endverbatim
  
 */
/*-------------------------------------------------------------------------*/
char * gnuplot_get_program_path(char * pname)
{
    int         i, j, lg;
    char    *   path;
    static char buf[PATH_MAXNAMESZ];

    /* Trivial case: try in CWD */
    sprintf(buf, "./%s", pname) ;
    if (access(buf, X_OK)==0) {
        sprintf(buf, ".");
        return buf ;
    }
    /* Try out in all paths given in the PATH variable */
    buf[0] = 0;
    path = getenv("PATH") ;
    if (path!=NULL) {
        for (i=0; path[i]; ) {
            for (j=i ; (path[j]) && (path[j]!=':') ; j++);
            lg = j - i;
            strncpy(buf, path + i, lg);
            if (lg == 0) buf[lg++] = '.';
            buf[lg++] = '/';
            strcpy(buf + lg, pname);
            if (access(buf, X_OK) == 0) {
                /* Found it! */
                break ;
            }
            buf[0] = 0;
            i = j;
            if (path[i] == ':') i++ ;
        }
    } else {
		fprintf(stderr, "PATH variable not set\n");
	}
    /* If the buffer is still empty, the command was not found */
    if (buf[0] == 0) return NULL ;
    /* Otherwise truncate the command name to yield path only */
    lg = strlen(buf) - 1 ;
    while (buf[lg]!='/') {
        buf[lg]=0 ;
        lg -- ;
    }
    buf[lg] = 0;
    return buf ;
}



/*-------------------------------------------------------------------------*/
/**
  @brief	Opens up a gnuplot session, ready to receive commands.
  @return	Newly allocated gnuplot control structure.

  This opens up a new gnuplot session, ready for input. The struct
  controlling a gnuplot session should remain opaque and only be
  accessed through the provided functions.

  The session must be closed using gnuplot_close().
 */
/*--------------------------------------------------------------------------*/

gnuplot_ctrl * gnuplot_init(void)
{
    gnuplot_ctrl *  handle ;

    if (getenv("DISPLAY") == NULL) {
        fprintf(stderr, "cannot find DISPLAY variable: is it set?\n") ;
    }
	if (gnuplot_get_program_path("gnuplot")==NULL) {
		fprintf(stderr, "cannot find gnuplot in your PATH");
		return NULL ;
	}

    /* 
     * Structure initialization:
     */
    handle = (gnuplot_ctrl*)malloc(sizeof(gnuplot_ctrl)) ;
    handle->nplots = 0 ;
    gnuplot_setstyle(handle, "points") ;
    handle->ntmp = 0 ;

    handle->nobj = 0;

    handle->gnucmd = popen("gnuplot", "w") ;



    if (handle->gnucmd == NULL) {
        fprintf(stderr, "error starting gnuplot\n") ;
        free(handle) ;
        return NULL ;
    }

    gnuplot_cmd(handle,"set obj 1 rectangle behind from screen 0,0 to screen 1,1");
    gnuplot_cmd(handle,"set obj 1 fillstyle solid 1.0 fillcolor rgbcolor \"white\"");
    gnuplot_cmd(handle,"set term x11");

    return handle;
}


/*-------------------------------------------------------------------------*/
/**
  @brief	Closes a gnuplot session previously opened by gnuplot_init()
  @param	handle Gnuplot session control handle.
  @return	void

  Kills the child PID and deletes all opened temporary files.
  It is mandatory to call this function to close the handle, otherwise
  temporary files are not cleaned and child process might survive.

 */
/*--------------------------------------------------------------------------*/

void gnuplot_close(gnuplot_ctrl * handle)
{
    int     i ;
	
    if (pclose(handle->gnucmd) == -1) {
        fprintf(stderr, "problem closing communication to gnuplot\n") ;
        return ;
    }
    if (handle->ntmp) {
        for (i=0 ; i<handle->ntmp ; i++) {
            remove(handle->to_delete[i]) ;
        }
    }
    free(handle) ;
    return ;
}


/*-------------------------------------------------------------------------*/
/**
  @brief	Sends a command to an active gnuplot session.
  @param	handle Gnuplot session control handle
  @param	cmd    Command to send, same as a printf statement.

  This sends a string to an active gnuplot session, to be executed.
  There is strictly no way to know if the command has been
  successfully executed or not.
  The command syntax is the same as printf.

  Examples:

  @code
  gnuplot_cmd(g, "plot %d*x", 23.0);
  gnuplot_cmd(g, "plot %g * cos(%g * x)", 32.0, -3.0);
  @endcode

  Since the communication to the gnuplot process is run through
  a standard Unix pipe, it is only unidirectional. This means that
  it is not possible for this interface to query an error status
  back from gnuplot.
 */
/*--------------------------------------------------------------------------*/

void gnuplot_cmd(gnuplot_ctrl *  handle, char *  cmd, ...)
{
    va_list ap ;
    char    local_cmd[GP_CMD_SIZE];

    va_start(ap, cmd);
    vsprintf(local_cmd, cmd, ap);
    va_end(ap);

    strcat(local_cmd, "\n");

    fputs(local_cmd, handle->gnucmd) ;
    fflush(handle->gnucmd) ;
    return ;
}


/*-------------------------------------------------------------------------*/
/**
  @brief	Change the plotting style of a gnuplot session.
  @param	h Gnuplot session control handle
  @param	plot_style Plotting-style to use (character string)
  @return	void

  The provided plotting style is a character string. It must be one of
  the following:

  - lines
  - points
  - linespoints
  - impulses
  - dots
  - steps
  - errorbars
  - boxes
  - boxeserrorbars
 */
/*--------------------------------------------------------------------------*/

void gnuplot_setstyle(gnuplot_ctrl * h, char * plot_style) 
{
    if (strcmp(plot_style, "lines") &&
        strcmp(plot_style, "points") &&
        strcmp(plot_style, "linespoints") &&
        strcmp(plot_style, "impulses") &&
        strcmp(plot_style, "dots") &&
        strcmp(plot_style, "steps") &&
        strcmp(plot_style, "errorbars") &&
        strcmp(plot_style, "boxes") &&
        strcmp(plot_style, "boxerrorbars")) {
        fprintf(stderr, "warning: unknown requested style: using points\n") ;
        strcpy(h->pstyle, "points") ;
    } else {
        strcpy(h->pstyle, plot_style) ;
    }
    return ;
}


/*-------------------------------------------------------------------------*/
/**
  @brief	Sets the x label of a gnuplot session.
  @param	h Gnuplot session control handle.
  @param	label Character string to use for X label.
  @return	void

  Sets the x label for a gnuplot session.
 */
/*--------------------------------------------------------------------------*/

void gnuplot_set_xlabel(gnuplot_ctrl * h, char * label)
{
    char    cmd[GP_CMD_SIZE] ;

    sprintf(cmd, "set xlabel \"%s\"", label) ;
    gnuplot_cmd(h, cmd) ;
    return ;
}


/*-------------------------------------------------------------------------*/
/**
  @brief	Sets the y label of a gnuplot session.
  @param	h Gnuplot session control handle.
  @param	label Character string to use for Y label.
  @return	void

  Sets the y label for a gnuplot session.
 */
/*--------------------------------------------------------------------------*/

void gnuplot_set_ylabel(gnuplot_ctrl * h, char * label)
{
    char    cmd[GP_CMD_SIZE] ;

    sprintf(cmd, "set ylabel \"%s\"", label) ;
    gnuplot_cmd(h, cmd) ;
    return ;
}


/*-------------------------------------------------------------------------*/
/**
  @brief	Resets a gnuplot session (next plot will erase previous ones).
  @param	h Gnuplot session control handle.
  @return	void

  Resets a gnuplot session, i.e. the next plot will erase all previous
  ones.
 */
/*--------------------------------------------------------------------------*/

void gnuplot_resetplot(gnuplot_ctrl * h)
{
    int     i ;
    if (h->ntmp) {
        for (i=0 ; i<h->ntmp ; i++) {
            remove(h->to_delete[i]) ;
        }
    }
    //gnuplot_cmd(h,"clear");
    h->ntmp = 0 ;
    h->nplots = 0 ;
    return ;
}



/*-------------------------------------------------------------------------*/
/**
  @brief	Plots a 2d graph from a list of doubles.
  @param	handle	Gnuplot session control handle.
  @param	d		Array of doubles.
  @param	n		Number of values in the passed array.
  @param	title	Title of the plot.
  @return	void

  Plots out a 2d graph from a list of doubles. The x-coordinate is the
  index of the float in the list, the y coordinate is the float in
  the list.

  Example:

  @code
    gnuplot_ctrl    *h ;
    float          d[50] ;
    int             i ;

    h = gnuplot_init() ;
    for (i=0 ; i<50 ; i++) {
        d[i] = (float)(i*i) ;
    }
    gnuplot_plot_x(h, d, 50, "parabola") ;
    sleep(2) ;
    gnuplot_close(h) ;
  @endcode
 */
/*--------------------------------------------------------------------------*/
template<typename T>
void gnuplot_plot_x_t(
    gnuplot_ctrl    *   handle,
    T          *   d,
    int                 n,
    const char            *   title
)
{
    int     i ;
	int		tmpfd ;
    char    name[128] ;
    char    cmd[GP_CMD_SIZE] ;
    char    line[GP_CMD_SIZE] ;

    int nmax = 5000;
    int dn = (n + nmax - 1)/nmax;


	if (handle==NULL || d==NULL || (n<1)) return ;

    /* Open one more temporary file? */
    if (handle->ntmp == GP_MAX_TMP_FILES - 1) {
        fprintf(stderr,
                "maximum # of temporary files reached (%d): cannot open more",
                GP_MAX_TMP_FILES) ;
        return ;
    }

    /* Open temporary file for output   */
	sprintf(name, "%s/gnuplot-i-XXXXXX", P_tmpdir);
    if ((tmpfd=mkstemp(name))==-1) {
        fprintf(stderr,"cannot create temporary file: exiting plot") ;
        return ;
    }

    /* Store file name in array for future deletion */
    strcpy(handle->to_delete[handle->ntmp], name) ;
    handle->ntmp ++ ;
    /* Write data to this file  */
    for (i=0 ; i<n ; i+=dn) {
		sprintf(line, "%g\n", d[i]);
		write(tmpfd, line, strlen(line));
    }
    close(tmpfd) ;

    /* Command to be sent to gnuplot    */
    if (handle->nplots > 0) {
        strcpy(cmd, "replot") ;
    } else {
        strcpy(cmd, "plot") ;
    }
    
    if (title == NULL) {
        sprintf(line, "%s \"%s\" with %s notitle", cmd, name, handle->pstyle) ;
    } else {
        sprintf(line, "%s \"%s\" title \"%s\" with %s", cmd, name,
                      title, handle->pstyle) ;
    }

    /* send command to gnuplot  */
    gnuplot_cmd(handle, line) ;
    handle->nplots++ ;
    return ;
}

void gnuplot_plot_x(
	    gnuplot_ctrl    *   handle,
	    float          *   d,
	    int                 n,
	    const char            *   title)
{
	gnuplot_plot_x_t<float>(handle,d,n,title);
}

void gnuplot_plot_x(
	    gnuplot_ctrl    *   handle,
	    double          *   d,
	    int                 n,
	    const char            *   title)
{
	gnuplot_plot_x_t<double>(handle,d,n,title);
}



/*-------------------------------------------------------------------------*/
/**
  @brief	Plot a 2d graph from a list of points.
  @param	handle		Gnuplot session control handle.
  @param	x			Pointer to a list of x coordinates.
  @param	y			Pointer to a list of y coordinates.
  @param	n			Number of doubles in x (assumed the same as in y).
  @param	title		Title of the plot.
  @return	void

  Plots out a 2d graph from a list of points. Provide points through a list
  of x and a list of y coordinates. Both provided arrays are assumed to
  contain the same number of values.

  @code
    gnuplot_ctrl    *h ;
	float			x[50] ;
	float			y[50] ;
    int             i ;

    h = gnuplot_init() ;
    for (i=0 ; i<50 ; i++) {
        x[i] = (float)(i)/10.0 ;
        y[i] = x[i] * x[i] ;
    }
    gnuplot_plot_xy(h, x, y, 50, "parabola") ;
    sleep(2) ;
    gnuplot_close(h) ;
  @endcode
 */
/*--------------------------------------------------------------------------*/
template<typename T>
void gnuplot_plot_xy_t(
    gnuplot_ctrl    *   handle,
	T			*	x,
	T			*	y,
    int                 n,
    const char            *   title,
    char* extras)
{
    int     i ;
	int		tmpfd ;
    char    name[128] ;
    char    cmd[GP_CMD_SIZE] ;
    char    line[GP_CMD_SIZE] ;

    int nmax = 4096;
    int dn = (n + nmax - 1)/nmax;

	if (handle==NULL || x==NULL || y==NULL || (n<1)) return ;

    /* Open one more temporary file? */
    if (handle->ntmp == GP_MAX_TMP_FILES - 1) {
        fprintf(stderr,
                "maximum # of temporary files reached (%d): cannot open more",
                GP_MAX_TMP_FILES) ;
        return ;
    }

    /* Open temporary file for output   */
	sprintf(name, "%s/gnuplot-i-XXXXXX", P_tmpdir);
    if ((tmpfd=mkstemp(name))==-1) {
        fprintf(stderr,"cannot create temporary file: exiting plot") ;
        return ;
    }
    /* Store file name in array for future deletion */
    strcpy(handle->to_delete[handle->ntmp], name) ;
    handle->ntmp ++ ;

    /* Write data to this file  */
    for (i=0 ; i<n; i+=dn) {
        sprintf(line, "%30.28g %30.28g\n", x[i], y[i]) ;
		write(tmpfd, line, strlen(line));
    }
    close(tmpfd) ;

    /* Command to be sent to gnuplot    */
    if (handle->nplots > 0) {
        strcpy(cmd, "replot") ;
    } else {
        strcpy(cmd, "plot") ;
    }
    
    if (title == NULL) {
        sprintf(line, "%s \"%s\"  with %s notitle %s ", cmd, name, handle->pstyle,extras) ;
    } else {
        sprintf(line, "%s \"%s\" title \"%s\" with %s %s", cmd, name,
                      title, handle->pstyle,extras) ;
    }

    /* send command to gnuplot  */
    gnuplot_cmd(handle, line) ;
    handle->nplots++ ;
    return ;
}

void gnuplot_plot_xy(
    gnuplot_ctrl    *   handle,
	float			*	x,
	float			*	y,
    int                 n,
    const char            *   title,
    char* extras)
{
	gnuplot_plot_xy_t<float>(handle,x,y,n,title,extras);

}

void gnuplot_plot_xy(
    gnuplot_ctrl    *   handle,
	double			*	x,
	double			*	y,
    int                 n,
    const char            *   title,
    char* extras)
{
	gnuplot_plot_xy_t<double>(handle,x,y,n,title,extras);
}

/*-------------------------------------------------------------------------*/
/**
  @brief	Plot a 2d graph from a list of points.
  @param	handle		Gnuplot session control handle.
  @param	x			Pointer to a list of x coordinates.
  @param	y			Pointer to a list of y coordinates.
  @param	n			Number of doubles in x (assumed the same as in y).
  @param	title		Title of the plot.
  @return	void

  Plots out a 2d graph from a list of points. Provide points through a list
  of x and a list of y coordinates. Both provided arrays are assumed to
  contain the same number of values.

  @code
    gnuplot_ctrl    *h ;
	float			x[50] ;
	float			y[50] ;
    int             i ;

    h = gnuplot_init() ;
    for (i=0 ; i<50 ; i++) {
        x[i] = (float)(i)/10.0 ;
        y[i] = x[i] * x[i] ;
    }
    gnuplot_plot_xy(h, x, y, 50, "parabola") ;
    sleep(2) ;
    gnuplot_close(h) ;
  @endcode
 */
/*--------------------------------------------------------------------------*/

void gnuplot_plot_xy_circles(
    gnuplot_ctrl    *   handle,
	float			*	x,
	float			*	y,
	float			*   radius,
    int                 n,
    const char            *   title,
    char				* extras = "")
{
    int     i ;
	int		tmpfd ;
    char    name[128] ;
    char    cmd[GP_CMD_SIZE] ;
    char    line[GP_CMD_SIZE] ;

	if (handle==NULL || x==NULL || y==NULL || (n<1)) return ;

    /* Open one more temporary file? */
    if (handle->ntmp == GP_MAX_TMP_FILES - 1) {
        fprintf(stderr,
                "maximum # of temporary files reached (%d): cannot open more",
                GP_MAX_TMP_FILES) ;
        return ;
    }

    /* Open temporary file for output   */
	sprintf(name, "%s/gnuplot-i-XXXXXX", P_tmpdir);
    if ((tmpfd=mkstemp(name))==-1) {
        fprintf(stderr,"cannot create temporary file: exiting plot") ;
        return ;
    }
    /* Store file name in array for future deletion */
    strcpy(handle->to_delete[handle->ntmp], name) ;
    handle->ntmp ++ ;

    /* Write data to this file  */
    for (i=0 ; i<n; i++) {
        sprintf(line, "%g %g %g\n", x[i], y[i],radius[i]) ;
		write(tmpfd, line, strlen(line));
    }
    close(tmpfd) ;

    /* Command to be sent to gnuplot    */
    if (handle->nplots > 0) {
        strcpy(cmd, "replot") ;
    } else {
        strcpy(cmd, "plot") ;
    }

    if (title == NULL) {
        sprintf(line, "%s \"%s\" notitle with circles %s", cmd, name,extras) ;
    } else {
        sprintf(line, "%s \"%s\" title \"%s\" with circles %s", cmd, name,
                      title,extras) ;
    }

    /* send command to gnuplot  */
    gnuplot_cmd(handle, "set style fill solid noborder") ;
    gnuplot_cmd(handle, line) ;
    handle->nplots++ ;
    return ;
}

/*-------------------------------------------------------------------------*/
/**
  @brief	Open a new session, plot a signal, close the session.
  @param	title	Plot title
  @param	style	Plot style
  @param	label_x	Label for X
  @param	label_y	Label for Y
  @param	x		Array of X coordinates
  @param	y		Array of Y coordinates (can be NULL)
  @param	n		Number of values in x and y.
  @return

  This function opens a new gnuplot session, plots the provided
  signal as an X or XY signal depending on a provided y, waits for
  a carriage return on stdin and closes the session.

  It is Ok to provide an empty title, empty style, or empty labels for
  X and Y. Defaults are provided in this case.
 */
/*--------------------------------------------------------------------------*/

void gnuplot_plot_once(
	const char	*	title,
	char	*	style,
	char	*	label_x,
	char	*	label_y,
	float	*	x,
	float	*	y,
	int			n
)
{
	gnuplot_ctrl	*	handle ;

	if (x==NULL || n<1) return ;

	if ((handle = gnuplot_init()) == NULL) return ;
	if (style!=NULL) {
		gnuplot_setstyle(handle, style);
	} else {
		gnuplot_setstyle(handle, "lines");
	}
	if (label_x!=NULL) {
		gnuplot_set_xlabel(handle, label_x);
	} else {
		gnuplot_set_xlabel(handle, "X");
	}
	if (label_y!=NULL) {
		gnuplot_set_ylabel(handle, label_y);
	} else {
		gnuplot_set_ylabel(handle, "Y");
	}
	if (y==NULL) {
		gnuplot_plot_x(handle, x, n, title);
	} else {
		gnuplot_plot_xy(handle, x, y, n, title);
	}
	printf("press ENTER to continue\n");
	while (getchar()!='\n') {}
	gnuplot_close(handle);
	return ;
}




/*-------------------------------------------------------------------------*/
/**
  @brief	Plot a slope on a gnuplot session.
  @param	handle		Gnuplot session control handle.
  @param	a			Slope.
  @param	b			Intercept.
  @param	title		Title of the plot.
  @return	void

  Plot a slope on a gnuplot session. The provided slope has an
  equation of the form y=ax+b

  Example:

  @code
    gnuplot_ctrl    *   h ;
    float              a, b ;

    h = gnuplot_init() ;
    gnuplot_plot_slope(h, 1.0, 0.0, "unity slope") ;
    sleep(2) ;
    gnuplot_close(h) ;
  @endcode
 */
/*--------------------------------------------------------------------------*/
    

void gnuplot_plot_slope(
    gnuplot_ctrl    *   handle,
    float              a,
    float              b,
    const char            *   title
)
{
    char    stitle[GP_TITLE_SIZE] ;
    char    cmd[GP_CMD_SIZE] ;

    if (title == NULL) {
        strcpy(stitle, "no title") ;
    } else {
        strcpy(stitle, title) ;
    }

    if (handle->nplots > 0) {
        sprintf(cmd, "replot %g * x + %g title \"%s\" with %s",
                      a, b, title, handle->pstyle) ;
    } else {
        sprintf(cmd, "plot %g * x + %g title \"%s\" with %s",
                      a, b, title, handle->pstyle) ;
    }
    gnuplot_cmd(handle, cmd) ;
    handle->nplots++ ;
    return ;
}


/*-------------------------------------------------------------------------*/
/**
  @brief	Plot a curve of given equation y=f(x).
  @param	h			Gnuplot session control handle.
  @param	equation	Equation to plot.
  @param	title		Title of the plot.
  @return	void

  Plots out a curve of given equation. The general form of the
  equation is y=f(x), you only provide the f(x) side of the equation.

  Example:

  @code
        gnuplot_ctrl    *h ;
        char            eq[80] ;

        h = gnuplot_init() ;
        strcpy(eq, "sin(x) * cos(2*x)") ;
        gnuplot_plot_equation(h, eq, "sine wave", normal) ;
        gnuplot_close(h) ;
  @endcode
 */
/*--------------------------------------------------------------------------*/

void gnuplot_plot_equation(
    gnuplot_ctrl    *   h,
    char            *   equation,
    const char            *   title
)
{
    char    cmd[GP_CMD_SIZE];
    char    plot_str[GP_EQ_SIZE] ;
    char    title_str[GP_TITLE_SIZE] ;

    if (title == NULL) {
        strcpy(title_str, "no title") ;
    } else {
        strcpy(title_str, title) ;
    }
    if (h->nplots > 0) {
        strcpy(plot_str, "replot") ;
    } else {
        strcpy(plot_str, "plot") ;
    }

    sprintf(cmd, "%s %s title \"%s\" with %s", 
                  plot_str, equation, title_str, h->pstyle) ;
    gnuplot_cmd(h, cmd) ;
    h->nplots++ ;
    return ;
}

/* vim: set ts=4 et sw=4 tw=75 */


void gnuplot_plot_xyz(
    gnuplot_ctrl    *   handle,
	float			*	x,
	float			*	y,
	float			*	z,
    int                 nx,
    int                 ny,
    const char            *   title
)
{
    int     i ;
	int		tmpfd ;
    char    name[128] ;
    char    cmd[GP_CMD_SIZE] ;
    char    line[GP_CMD_SIZE] ;

    int max_points = 101;

    int nx_inc = (nx+max_points-1)/max_points;
    int ny_inc = (ny+max_points-1)/max_points;

	if (handle==NULL || x==NULL || y==NULL || (nx<1) || (ny<1)) return ;

    /* Open one more temporary file? */
    if (handle->ntmp == GP_MAX_TMP_FILES - 1) {
        fprintf(stderr,
                "maximum # of temporary files reached (%d): cannot open more",
                GP_MAX_TMP_FILES) ;
        return ;
    }

    /* Open temporary file for output   */
	sprintf(name, "%s/gnuplot-i-XXXXXX", P_tmpdir);
    if ((tmpfd=mkstemp(name))==-1) {
        fprintf(stderr,"cannot create temporary file: exiting plot") ;
        return ;
    }
    /* Store file name in array for future deletion */
    strcpy(handle->to_delete[handle->ntmp], name) ;
    handle->ntmp ++ ;

    /* Write data to this file  */
    for (i=0 ; i<=ny+ny_inc; i+=ny_inc) {
    	for(int j=0;j<=nx+nx_inc;j+=nx_inc)
    	{
    		int it = fmin(i,ny-1);
    		int jt = fmin(j,nx-1);
    		sprintf(line, "%g %g %g\n", x[jt], y[it], z[jt+nx*it]) ;
			write(tmpfd, line, strlen(line));
    	}
		sprintf(line, "\n") ;
		write(tmpfd, line, strlen(line));
    }
    close(tmpfd) ;

    /* Command to be sent to gnuplot    */
    if (handle->nplots > 0) {
        strcpy(cmd, "replot") ;
    } else {
    	gnuplot_cmd(handle,"set iso 10");
    	gnuplot_cmd(handle,"set samp 10");
//    	gnuplot_cmd(handle,"set ztics 1");
    	gnuplot_cmd(handle,"unset key");
		gnuplot_cmd(handle,"set contour");
		gnuplot_cmd(handle,"set cntrparam levels 8");
		gnuplot_cmd(handle,"set hidd");
		//strcpy(cmd, "dgrid3d") ;
        strcpy(cmd, "splot") ;
    }

    if (title == NULL) {
        sprintf(line, "%s \"%s\" with pm3d", cmd, name) ;
    } else {
        sprintf(line, "%s \"%s\" title \"%s\" with pm3d", cmd, name,
                      title) ;
    }

    /* send command to gnuplot  */
    gnuplot_cmd(handle, line) ;
    handle->nplots++ ;
    return ;
}

void gnuplot_plot_vector(
    gnuplot_ctrl    *   handle,
	float			*	x,
	float			*	y,
	float			* 	z,
	float			*	dx,
	float			*	dy,
    int                 nx,
    int                 ny,
    const char            *   title
)
{
    int     i ;
	int		tmpfd ;
    char    name[128] ;
    char    cmd[GP_CMD_SIZE] ;
    char    line[GP_CMD_SIZE] ;

    int max_vecs = 20;

    int nx_inc = (nx+max_vecs-1)/max_vecs;
    int ny_inc = (ny+max_vecs-1)/max_vecs;
    float scale = sqrt(nx_inc*ny_inc);

	if (handle==NULL || x==NULL || y==NULL || (nx<1) || (ny<1)) return ;

    /* Open one more temporary file? */
    if (handle->ntmp == GP_MAX_TMP_FILES - 1) {
        fprintf(stderr,
                "maximum # of temporary files reached (%d): cannot open more",
                GP_MAX_TMP_FILES) ;
        return ;
    }

    /* Open temporary file for output   */
	sprintf(name, "%s/gnuplot-i-XXXXXX", P_tmpdir);
    if ((tmpfd=mkstemp(name))==-1) {
        fprintf(stderr,"cannot create temporary file: exiting plot") ;
        return ;
    }
    /* Store file name in array for future deletion */
    strcpy(handle->to_delete[handle->ntmp], name) ;
    handle->ntmp ++ ;

    /* Write data to this file  */
    for (i=0 ; i<ny-ny_inc; i+=ny_inc) {
    	for(int j=0;j<nx-nx_inc;j+=nx_inc)
    	{
    		int ip,jp;
    		ip = fmin(i+ny_inc,ny-1);
    		jp = fmin(j+nx_inc,nx-1);
    		float xt,yt,zt;
    		float dxt,dyt,dzt;
    		xt = (x[j]+x[jp])/2.0;
    		yt = (y[i]+y[ip])/2.0;
    		zt = (z[j+nx*i]+z[jp+nx*ip])/2.0;
    		dxt = (dx[j+nx*i]+dx[jp+nx*ip])/2.0;
    		dyt = (dy[j+nx*i]+dy[jp+nx*ip])/2.0;
    		dzt = 0;
    		sprintf(line, "%g %g %g %g %g %g\n", xt, yt, zt, dxt*scale,dyt*scale,dzt*scale);
			write(tmpfd, line, strlen(line));
    	}
		sprintf(line, "\n") ;
		write(tmpfd, line, strlen(line));
    }
    close(tmpfd) ;

    /* Command to be sent to gnuplot    */
    if (handle->nplots > 0) {
        strcpy(cmd, "replot") ;
    } else {
    	//strcpy(cmd, "dgrid3d") ;
        strcpy(cmd, "splot") ;
    }

    if (title == NULL) {
        sprintf(line, "%s \"%s\" with pm3d", cmd, name) ;
    } else {
        sprintf(line, "%s \"%s\" title \"%s\" w vec size  10, 15 filled", cmd, name,
                      title) ;
    }

    /* send command to gnuplot  */
    gnuplot_cmd(handle, line) ;
    handle->nplots++ ;
    return ;
}

void gnuplot_plot_vector3D(
    gnuplot_ctrl    *   handle,
	float			*	x,
	float			*	y,
	float			* 	z,
	float			*	dx,
	float			*	dy,
	float			*	dz,
    int                 nx,
    int                 ny,
    const char            *   title
)
{
    int     i ;
	int		tmpfd ;
    char    name[128] ;
    char    cmd[GP_CMD_SIZE] ;
    char    line[GP_CMD_SIZE] ;

    int max_vecs = 20;

    int nx_inc = (nx+max_vecs-1)/max_vecs;
    int ny_inc = (ny+max_vecs-1)/max_vecs;

    float scale = sqrt(nx_inc*ny_inc);


	if (handle==NULL || x==NULL || y==NULL || (nx<1) || (ny<1)) return ;

    /* Open one more temporary file? */
    if (handle->ntmp == GP_MAX_TMP_FILES - 1) {
        fprintf(stderr,
                "maximum # of temporary files reached (%d): cannot open more",
                GP_MAX_TMP_FILES) ;
        return ;
    }

    /* Open temporary file for output   */
	sprintf(name, "%s/gnuplot-i-XXXXXX", P_tmpdir);
    if ((tmpfd=mkstemp(name))==-1) {
        fprintf(stderr,"cannot create temporary file: exiting plot") ;
        return ;
    }
    /* Store file name in array for future deletion */
    strcpy(handle->to_delete[handle->ntmp], name) ;
    handle->ntmp ++ ;

    /* Write data to this file  */
    for (i=0 ; i<ny-ny_inc; i+=ny_inc) {
    	for(int j=0;j<nx-nx_inc;j+=nx_inc)
    	{
    		int ip,jp;
    		ip = fmin(i+ny_inc,ny-1);
    		jp = fmin(j+nx_inc,nx-1);
    		float xt,yt,zt;
    		float dxt,dyt,dzt;
    		xt = (x[j]+x[jp])/2.0;
    		yt = (y[i]+y[ip])/2.0;
    		zt = (z[j+nx*i]+z[jp+nx*ip])/2.0;
    		dxt = (dx[j+nx*i]+dx[jp+nx*ip])/2.0;
    		dyt = (dy[j+nx*i]+dy[jp+nx*ip])/2.0;
    		dzt = (dz[j+nx*i]+dz[jp+nx*ip])/2.0;
    		sprintf(line, "%20g %20g %20g %20g %20g %20g\n", xt, yt, zt, dxt*scale,dyt*scale,dzt*scale);
			write(tmpfd, line, strlen(line));
    	}
		sprintf(line, "\n") ;
		write(tmpfd, line, strlen(line));
    }
    close(tmpfd) ;

    /* Command to be sent to gnuplot    */
    if (handle->nplots > 0) {
        strcpy(cmd, "replot") ;
    } else {
    	//strcpy(cmd, "dgrid3d") ;
        strcpy(cmd, "splot") ;
    }

    if (title == NULL) {
        sprintf(line, "%s \"%s\" with pm3d", cmd, name) ;
    } else {
        sprintf(line, "%s \"%s\" title \"%s\" w vec size  10, 15 filled", cmd, name,
                      title) ;
    }

    /* send command to gnuplot  */
    gnuplot_cmd(handle, line) ;
    handle->nplots++ ;
    return ;
}



void gnuplot_plot_rbgaimage(
    gnuplot_ctrl    *   handle,
	pixel			*	pixels,
	int 				nx,
    int               ny,
    float			xmin,
    float 			xmax,
    float			ymin,
    float 			ymax,
    const char            *   title
)
{
    int     i ;
	int		tmpfd ;
    char    name[128] ;
    char    cmd[GP_CMD_SIZE] ;
    char    line[GP_CMD_SIZE] ;

    float dx = (xmax-xmin)/nx;
    float dy = (ymax-ymin)/ny;

	if (handle==NULL || pixels==NULL ||  (nx<1) ||  (ny<1))return ;

    /* Open one more temporary file? */
    if (handle->ntmp == GP_MAX_TMP_FILES - 1) {
        fprintf(stderr,
                "maximum # of temporary files reached (%d): cannot open more",
                GP_MAX_TMP_FILES) ;
        return ;
    }

    /* Open temporary file for output   */
	sprintf(name, "%s/gnuplot-i-XXXXXX", P_tmpdir);
    if ((tmpfd=mkstemp(name))==-1) {
        fprintf(stderr,"cannot create temporary file: exiting plot") ;
        return ;
    }
    /* Store file name in array for future deletion */
    strcpy(handle->to_delete[handle->ntmp], name) ;
    handle->ntmp ++ ;

    gnuplot_cmd(handle,"set xrange [%f:%f]",xmin,xmax);
    gnuplot_cmd(handle,"set yrange [%f:%f]",ymin,ymax);
    gnuplot_cmd(handle,"set size square");


    /* Write data to this file  */
    for (i=0 ; i<nx*ny; i++) {

    	int ix = i%nx;
    	int iy = i/nx;
    	float xt = ix*dx+xmin;
    	float yt = iy*dy+ymin;
    	pixel my_pixel = pixels[i];

        sprintf(line, "%g %g %i %i %i %i\n", xt, yt, my_pixel.r,my_pixel.g,my_pixel.b,my_pixel.a) ;
		write(tmpfd, line, strlen(line));

    }
    close(tmpfd) ;

    /* Command to be sent to gnuplot    */
    if (handle->nplots > 0) {
        strcpy(cmd, "replot") ;
    } else {
        strcpy(cmd, "plot") ;
    }

    if (title == NULL) {
        sprintf(line, "%s \"%s\" with rgbalpha failsafe ", cmd, name) ;
    } else {
        sprintf(line, "%s \"%s\" title \"%s\" with rgbalpha", cmd, name,
                      title) ;
    }

    /* send command to gnuplot  */
    gnuplot_cmd(handle, line) ;
    handle->nplots++ ;
    return ;
}


void gnuplot_setup_cell(
	gnuplot_ctrl	* 	handle,
	cell	*				my_cell
)
{
	handle->nobj += 1;

	// Set the cell to be a polygon with the following vertices
	gnuplot_cmd(handle,"set object %i polygon from "
			"%f, %f to"
			" %f, %f to"
			" %f, %f to"
			" %f, %f to"
			" %f, %f",
			handle->nobj,
			my_cell->x[0],my_cell->y[0],
			my_cell->x[1],my_cell->y[1],
			my_cell->x[2],my_cell->y[2],
			my_cell->x[3],my_cell->y[3],
			my_cell->x[0],my_cell->y[0]);

	// Set the line color
	gnuplot_cmd(handle,"set object %i lw 1 front",handle->nobj);
}

void gnuplot_color_cell(
	gnuplot_ctrl	* 	handle,
	float 			 	value,
	int					icell
)
{

	// Set the line color
	gnuplot_cmd(handle,"set object %i fillcolor palette frac %f "
							"fillstyle solid border rgb \"black\" lw 1 front",icell,value);
}

/*-------------------------------------------------------------------------*/
/**
  @brief	Plot a 2d polygon from a list of points.
  @param	handle		Gnuplot session control handle.
  @param	x			Pointer to a list of x coordinates.
  @param	y			Pointer to a list of y coordinates.
  @param	n			Number of doubles in x (assumed the same as in y).
  @param	title		Title of the plot.
  @return	void

  Plots out a 2d graph from a list of points. Provide points through a list
  of x and a list of y coordinates. Both provided arrays are assumed to
  contain the same number of values.

  @code
    gnuplot_ctrl    *h ;
	quadralateral shapes[50] ;
	float			color[50] ;
    int             i ;

    h = gnuplot_init() ;
    for (i=0 ; i<50 ; i++) {
        x[i] = (float)(i)/10.0 ;
        y[i] = x[i] * x[i] ;
    }
    gnuplot_plot_xy(h, x, y, 50, "parabola") ;
    sleep(2) ;
    gnuplot_close(h) ;
  @endcode
 */
/*--------------------------------------------------------------------------*/


void gnuplot_setup_mesh(
    gnuplot_ctrl    *   handle,
    cell	 *			cells,
    float	* 			xdims,
    float	*			ydims,
    int                 ncells
)
{

	// Set pm3d so that our palette actually works
	gnuplot_cmd(handle,"set pm3d");

	// Set up the plot window dimensions
	gnuplot_cmd(handle,"set xrange [%f:%f]",xdims[0],xdims[1]);
	gnuplot_cmd(handle,"set yrange [%f:%f]",ydims[0],ydims[1]);


	for(int i=0;i<ncells;i++)
	{
		gnuplot_setup_cell(handle,cells+i);
	}

	gnuplot_cmd(handle,"plot 1 lw 0");

    return ;
}

void gnuplot_fill_mesh(
		gnuplot_ctrl    *   handle,
	    float			*		value,
	    int                 ncells,
	    float		minval = 0,
	    float		maxval = 10
	    )
{
	float bval = minval;
	float gval = 0.3*(maxval-minval)+minval;
	float yval = 0.6*(maxval-minval)+minval;
	float rval = maxval;

	// Set the color palette
	gnuplot_cmd(handle,"set palette defined ( %f \"blue\", %f \"green\","
			" %f \"yellow\", %f \"red\" )",bval,gval,yval,rval);


	for(int i=0;i<ncells;i++)
	{
		float temp = (value[i]-minval)/(maxval-minval);
		gnuplot_color_cell(handle,temp,i+1);
	}

	gnuplot_cmd(handle,"replot");

	return;
}

void gnuplot_save_pdf(
		gnuplot_ctrl*			handle,
		const char*				filename)
{
	char line[128];



//	gnuplot_cmd(handle,"set term epslatex color lw 10 size 11,8.5  font 20 ");
	gnuplot_cmd(handle,"set term pdf");

	sprintf(line,"set output \"%s.pdf\"",filename);

	gnuplot_cmd(handle,line);

	gnuplot_cmd(handle,"replot");

	gnuplot_cmd(handle,"set term x11");
	gnuplot_cmd(handle,"set out");
}



