
/*-------------------------------------------------------------------------*/
/**
  @file		gnuplot_i.h
  @author	N. Devillard, J. Payne
  @date		2011
  @version	$Revision: 2.0 $
  @brief	C interface to gnuplot.

  gnuplot is a freely available, command-driven graphical display tool for
  Unix. It compiles and works quite well on a number of Unix flavours as
  well as other operating systems. The following module enables sending
  display requests to gnuplot through simple C calls.
  
*/
/*--------------------------------------------------------------------------*/

/*
	$Id: gnuplot_i.h,v 2.00 2011  $
	$Author: ndevilla $
	$Date: 2011$
	$Revision: 2.0 $
 */

#ifndef _GNUPLOT_PIPES_H_
#define _GNUPLOT_PIPES_H_

/*---------------------------------------------------------------------------
                                Includes
 ---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdarg.h>


/** Maximal number of simultaneous temporary files */
#define GP_MAX_TMP_FILES    64
/** Maximal size of a temporary file name */
#define GP_TMP_NAME_SIZE    512

typedef struct cell
{
	float x[4];
	float y[4];
}cell;

/*---------------------------------------------------------------------------
                                New Types
 ---------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------*/
/**
  @typedef	gnuplot_ctrl
  @brief	gnuplot session handle (opaque type).

  This structure holds all necessary information to talk to a gnuplot
  session. It is built and returned by gnuplot_init() and later used
  by all functions in this module to communicate with the session, then
  meant to be closed by gnuplot_close().

  This structure is meant to remain opaque, you normally do not need
  to know what is contained in there.
 */
/*-------------------------------------------------------------------------*/

typedef struct _GNUPLOT_CTRL_ {
    /** Pipe to gnuplot process */
    FILE    * gnucmd ;
    
    /** Number of currently active plots */
    int       nplots ;
	/** Current plotting style */
    char      pstyle[32] ;

    /** Name of temporary files */
    char      to_delete[GP_MAX_TMP_FILES][GP_TMP_NAME_SIZE] ;
	/** Number of temporary files */
    int       ntmp ;

    /** Number of objects */
    int nobj;
} gnuplot_ctrl ;

/*---------------------------------------------------------------------------
                        Function ANSI C prototypes
 ---------------------------------------------------------------------------*/

typedef struct{
	int r,b,g,a;
}pixel;

/*-------------------------------------------------------------------------*/
/**
  @brief    Find out where a command lives in your PATH.
  @param    pname Name of the program to look for.
  @return   pointer to statically allocated character string.

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
char * gnuplot_get_program_path(char * pname);

/*-------------------------------------------------------------------------*/
/**
  @brief    Opens up a gnuplot session, ready to receive commands.
  @return   Newly allocated gnuplot control structure.

  This opens up a new gnuplot session, ready for input. The struct
  controlling a gnuplot session should remain opaque and only be
  accessed through the provided functions.

  The session must be closed using gnuplot_close().
 */
/*--------------------------------------------------------------------------*/
gnuplot_ctrl * gnuplot_init(void);

/*-------------------------------------------------------------------------*/
/**
  @brief    Closes a gnuplot session previously opened by gnuplot_init()
  @param    handle Gnuplot session control handle.
  @return   void

  Kills the child PID and deletes all opened temporary files.
  It is mandatory to call this function to close the handle, otherwise
  temporary files are not cleaned and child process might survive.

 */
/*--------------------------------------------------------------------------*/
void gnuplot_close(gnuplot_ctrl * handle);

/*-------------------------------------------------------------------------*/
/**
  @brief    Sends a command to an active gnuplot session.
  @param    handle Gnuplot session control handle
  @param    cmd    Command to send, same as a printf statement.

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
void gnuplot_cmd(gnuplot_ctrl *  handle, char *  cmd, ...);

/*-------------------------------------------------------------------------*/
/**
  @brief    Change the plotting style of a gnuplot session.
  @param    h Gnuplot session control handle
  @param    plot_style Plotting-style to use (character string)
  @return   void

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
void gnuplot_setstyle(gnuplot_ctrl * h, char * plot_style);

/*-------------------------------------------------------------------------*/
/**
  @brief    Sets the x label of a gnuplot session.
  @param    h Gnuplot session control handle.
  @param    label Character string to use for X label.
  @return   void

  Sets the x label for a gnuplot session.
 */
/*--------------------------------------------------------------------------*/
void gnuplot_set_xlabel(gnuplot_ctrl * h, char * label);


/*-------------------------------------------------------------------------*/
/**
  @brief    Sets the y label of a gnuplot session.
  @param    h Gnuplot session control handle.
  @param    label Character string to use for Y label.
  @return   void

  Sets the y label for a gnuplot session.
 */
/*--------------------------------------------------------------------------*/
void gnuplot_set_ylabel(gnuplot_ctrl * h, char * label);

/*-------------------------------------------------------------------------*/
/**
  @brief    Resets a gnuplot session (next plot will erase previous ones).
  @param    h Gnuplot session control handle.
  @return   void

  Resets a gnuplot session, i.e. the next plot will erase all previous
  ones.
 */
/*--------------------------------------------------------------------------*/
void gnuplot_resetplot(gnuplot_ctrl * h);


/*-------------------------------------------------------------------------*/
/**
  @brief    Plots a 2d graph from a list of doubles.
  @param    handle  Gnuplot session control handle.
  @param    d       Array of doubles.
  @param    n       Number of values in the passed array.
  @param    title   Title of the plot.
  @return   void

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
void gnuplot_plot_x(gnuplot_ctrl * handle, float * d, int n,const char * title = NULL);
void gnuplot_plot_x(gnuplot_ctrl * handle, double * d, int n,const char * title = NULL);

/*-------------------------------------------------------------------------*/
/**
  @brief    Plot a 2d graph from a list of points.
  @param    handle      Gnuplot session control handle.
  @param    x           Pointer to a list of x coordinates.
  @param    y           Pointer to a list of y coordinates.
  @param    n           Number of doubles in x (assumed the same as in y).
  @param    title       Title of the plot.
  @return   void

  Plots out a 2d graph from a list of points. Provide points through a list
  of x and a list of y coordinates. Both provided arrays are assumed to
  contain the same number of values.

  @code
    gnuplot_ctrl    *h ;
    float          x[50] ;
    float          y[50] ;
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
void gnuplot_plot_xy(
    gnuplot_ctrl    *   handle,
    float          *   x,
    float          *   y,
    int                 n,
    const char            *   title  = NULL,
    char				* extras  = ""
);

void gnuplot_plot_xy(
    gnuplot_ctrl    *   handle,
    double          *   x,
    double          *   y,
    int                 n,
    const char            *   title  = NULL,
    char				* extras  = ""
);

/*-------------------------------------------------------------------------*/
/**
  @brief    Plot a 2d graph from a list of points.
  @param    handle      Gnuplot session control handle.
  @param    x           Pointer to a list of x coordinates.
  @param    y           Pointer to a list of y coordinates.
  @param    n           Number of doubles in x (assumed the same as in y).
  @param    title       Title of the plot.
  @return   void

  Plots out a 2d graph from a list of points. Provide points through a list
  of x and a list of y coordinates. Both provided arrays are assumed to
  contain the same number of values.

  @code
    gnuplot_ctrl    *h ;
    float          x[50] ;
    float          y[50] ;
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
    float          *   x,
    float          *   y,
    float		  *   radius,
    int                 n,
    const char            *   title,
    char				* extras
) ;


/*-------------------------------------------------------------------------*/
/** 
  @brief    Open a new session, plot a signal, close the session.
  @param    title   Plot title
  @param    style   Plot style
  @param    label_x Label for X
  @param    label_y Label for Y
  @param    x       Array of X coordinates
  @param    y       Array of Y coordinates (can be NULL)
  @param    n       Number of values in x and y.
  @return

  This function opens a new gnuplot session, plots the provided
  signal as an X or XY signal depending on a provided y, waits for
  a carriage return on stdin and closes the session.

  It is Ok to provide an empty title, empty style, or empty labels for
  X and Y. Defaults are provided in this case.
 */
/*--------------------------------------------------------------------------*/
void gnuplot_plot_once(
    const char    *   title,
    char    *   style,
    char    *   label_x,
    char    *   label_y,
    float  *   x,
    float  *   y,
    int         n
);

/*-------------------------------------------------------------------------*/
/**
  @brief    Plot a slope on a gnuplot session.
  @param    handle      Gnuplot session control handle.
  @param    a           Slope.
  @param    b           Intercept.
  @param    title       Title of the plot.
  @return   void

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
) ;

/*-------------------------------------------------------------------------*/
/**
  @brief    Plot a curve of given equation y=f(x).
  @param    h           Gnuplot session control handle.
  @param    equation    Equation to plot.
  @param    title       Title of the plot.
  @return   void

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
void gnuplot_plot_equation(gnuplot_ctrl * h, char * equation,const char * title) ;

/*-------------------------------------------------------------------------*/
/**
  @brief    Plot a 2d graph from a list of points.
  @param    handle      Gnuplot session control handle.
  @param    x           Pointer to a list of x coordinates.
  @param    y           Pointer to a list of y coordinates.
  @param    z           Pointer to a list of y coordinates.
  @param    n           Number of doubles in x (assumed the same as in y).
  @param    title       Title of the plot.
  @return   void

  Plots out a 2d graph from a list of points. Provide points through a list
  of x and a list of y coordinates. Both provided arrays are assumed to
  contain the same number of values.

  @code
    gnuplot_ctrl    *h ;
    float          x[50] ;
    float          y[50] ;
    float 				z[50];
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
void gnuplot_plot_xyz(
    gnuplot_ctrl    *   handle,
    float          *   x,
    float          *   y,
    float 			  *   z,
    int                 nx,
    int					ny,
    const char            *   title
) ;

void gnuplot_plot_vector(
    gnuplot_ctrl    *   handle,
	float			*	x,
	float			*	y,
	float			*	z,
	float			*	dx,
	float			*	dy,
    int                 nx,
    int                 ny,
    const char            *   title
);

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
);

void gnuplot_plot_rbgaimage(
    gnuplot_ctrl    *   handle,
	pixel			*	pixels,
	int 				nx,
    int               ny,
    float			xmin,
    float 			xmax,
    float			ymin,
    float 			ymax,
    const char            *   title = ""
);

void gnuplot_setup_cell(
	gnuplot_ctrl	* 	handle,
	cell	*				my_cell
);

void gnuplot_color_cell(
	gnuplot_ctrl	* 	handle,
	float 			 	value,
	int					icell
);

void gnuplot_setup_mesh(
    gnuplot_ctrl    *   handle,
    cell	 *			cells,
    float	* 			xdims,
    float	*			ydims,
    int                 ncells
);

void gnuplot_fill_mesh(
		gnuplot_ctrl    *   handle,
	    float			*		value,
	    int                 ncells,
	    float		minval,
	    float		maxval
	    );

void gnuplot_save_pdf(
		gnuplot_ctrl*			handle,
		const char*				filename);

#endif
