//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Gergely Mátyás
// Neptun : X9VG6O
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================


#include "framework.h"


struct Hit {
	float t;
	vec3 position, normal;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Light {
	vec3 p;
	vec3 Le;
	Light(vec3 _p, vec3 _Le) {
		p = _p;
		Le = _Le;
	}
};


struct Camera {
	vec3 eye, lookat, right, up;

	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};


std::vector<std::string> split(const std::string& str, std::string delimiter) {
	std::vector<std::string> tokens;
	std::string token;
	std::size_t start = str.find_first_not_of('\t');
	std::size_t end = str.find(delimiter, start);
	while (end != std::string::npos) {
		token = str.substr(start, end - start);
		tokens.push_back(token);
		start = str.find_first_not_of(' ', end);		
		end = str.find(delimiter, start);
	}
	token = str.substr(start, end);
	tokens.push_back(token);
	return tokens;
}

void readObject(const std::string& input, std::vector<vec3>& vertices, std::vector<vec3>& pairings) {
	std::string line;
	std::size_t start = 0;
	std::size_t end = input.find('\n');
	while (end != std::string::npos) {
		line = input.substr(start, end - start);
		std::vector<std::string> tokens = split(line, " ");
		std::vector<float> floats;
		std::string mode = tokens.at(0);

		for (int i = 1; i < tokens.size(); ++i) {
			floats.push_back(std::stof(tokens.at(i)));
		}

		if (mode[0] == 'v')
			vertices.push_back(vec3(floats.at(0), floats.at(1), floats.at(2)));
		else if( mode[0] == 'f')
			pairings.push_back(vec3(floats.at(0), floats.at(1), floats.at(2)));

		start = end + 1;
		end = input.find('\n', start);
	}
}

struct Triangle : Intersectable {
	vec3 r1, r2, r3;
	vec3 normal;

	Triangle(vec3 r1, vec3 r2, vec3 r3) {
		this->r1 = r1;
		this->r2 = r2;
		this->r3 = r3;

		normal = normalize(cross(r2 - r1, r3 - r1));
	}

	void reverseNormal() {
		normal = -1 * normal;
	}

	Hit intersect(const Ray& ray) {
		Hit *hit = new Hit();
		
		float t = dot((r1 - ray.start), normal) / dot(ray.dir, normal);

		if (t < 0)
			{Hit ret = *hit; delete hit; return ret;}

		vec3 p = ray.start + ray.dir * t;

		float check1 = dot(cross(r2 - r1, p - r1), normal);
		float check2 = dot(cross(r3 - r2, p - r2), normal);
		float check3 = dot(cross(r1 - r3, p - r3), normal);

		if (check1 > 0 && check2 > 0 && check3 > 0) {
			hit->normal = normal;
			hit->position = p;
			hit->t = t;
		}

		{Hit ret = *hit; delete hit; return ret;}
		
	}
};

struct Cube : Intersectable {
	std::vector<Triangle*> triangles;

	Cube(float scale, float displaceX, float displaceY, float displaceZ) {
		
		std::string obj(
			R"(v  0.0  0.0  0.0
			v  0.0  0.0  1.0
			v  0.0  1.0  0.0
			v  0.0  1.0  1.0
			v  1.0  0.0  0.0
			v  1.0  0.0  1.0
			v  1.0  1.0  0.0
			v  1.0  1.0  1.0
			f  1  7  5
			f  1  3  7
			f  1  4  3
			f  1  2  4
			f  3  8  7
			f  3  4  8
			f  5  7  8
			f  5  8  6
			f  1  5  6
			f  1  6  2
			f  2  6  8
			f  2  8  4
			)"
		);
		
		std::vector<vec3> vert;
		std::vector<vec3> pair;
		readObject(obj, vert, pair);

		for (int i = 0; i < vert.size(); ++i) {
			vert[i].x += -0.5 + displaceX;
			vert[i].y += -0.5 + displaceY;
			vert[i].z += -0.5 + displaceZ;
		}
		
		for (vec3 p : pair) {
			triangles.push_back(new Triangle(scale * vert[p.x - 1], scale * vert[p.y - 1], scale * vert[p.z - 1]));
		}
		
		for (int i = 0; i < 4; ++i)
			triangles[i]->reverseNormal();
			
	}

	Hit intersect(const Ray& ray) {
		Hit* hit = new Hit();
	
		for (Triangle* triangle : triangles) {
			Hit temp = triangle->intersect(ray);
			if (hit->t < 0)
				*hit = temp;
			else if (temp.t > 0 && temp.t < hit->t)
				*hit = temp;
		}
		{Hit ret = *hit; delete hit; return ret;}
	}
};

struct Icosahedron : Intersectable {
	std::vector<Triangle*> triangles;

	Icosahedron(float scale, float displaceX, float displaceY, float displaceZ) {
		
		std::string obj(
			R"(v  0 -0.525731  0.850651
			v  0.850651  0  0.525731
			v  0.850651  0 -0.525731
			v -0.850651  0 -0.525731
			v -0.850651  0  0.525731
			v -0.525731  0.850651  0
			v  0.525731  0.850651  0
			v  0.525731 -0.850651  0
			v -0.525731 -0.850651  0
			v  0 -0.525731 -0.850651
			v  0  0.525731 -0.850651
			v  0  0.525731  0.850651
			f  2  3  7
			f  2  8  3
			f  4  5  6
			f  5  4  9
			f  7  6  12
			f  6  7  11
			f  10  11  3
			f  11  10  4
			f  8  9  10
			f  9  8  1
			f  12  1  2
			f  1  12  5
			f  7  3  11
			f  2  7  12
			f  4  6  11
			f  6  5  12
			f  3  8  10
			f  8  2  1
			f  4  10  9
			f  5  9  1
			)"
		);

		std::vector<vec3> vert;
		std::vector<vec3> pair;
		readObject(obj, vert, pair);

		for (int i = 0; i < vert.size(); ++i) {
			vert[i].x += -0.5 + displaceX;
			vert[i].y += -0.5 + displaceY;
			vert[i].z += -0.5 + displaceZ;
		}

		for (vec3 p : pair) {
			triangles.push_back(new Triangle(scale * vert[p.x - 1], scale * vert[p.z - 1], scale * vert[p.y - 1]));
		}
		
	}

	Hit intersect(const Ray& ray) {
		Hit* hit = new Hit();

		for (Triangle* triangle : triangles) {
			Hit temp = triangle->intersect(ray);
			if (hit->t < 0)
				*hit = temp;
			else if (temp.t > 0 && temp.t < hit->t)
				*hit = temp;
		}
		{Hit ret = *hit; delete hit; return ret;}
	}

};



struct Octahedron : Intersectable {
	std::vector<Triangle*> triangles;

	Octahedron(float scale, float displaceX, float displaceY, float displaceZ) {
		std::string obj(
			R"(v  1  0  0
			v  0 -1  0
			v -1  0  0
			v  0  1  0
			v  0  0  1
			v  0  0 -1
			f  2  1  5
			f  3  2  5
			f  4  3  5
			f  1  4  5
			f  1  2  6
			f  2  3  6
			f  3  4  6
			f  4  1  6
			)"
		);

		std::vector<vec3> vert;
		std::vector<vec3> pair;
		readObject(obj, vert, pair);

		for (int i = 0; i < vert.size(); ++i) {
			vert[i].x += -0.5 + displaceX;
			vert[i].y += -0.5 + displaceY;
			vert[i].z += -0.5 + displaceZ;
		}

		for (vec3 p : pair) {
			triangles.push_back(new Triangle(scale * vert[p.x - 1], scale * vert[p.z - 1], scale * vert[p.y - 1]));
		}
	}

	Hit intersect(const Ray& ray) {
		Hit* hit = new Hit();

		for (Triangle* triangle : triangles) {
			Hit temp = triangle->intersect(ray);
			if (hit->t < 0)
				*hit = temp;
			else if (temp.t > 0 && temp.t < hit->t)
				*hit = temp;
		}
		{Hit ret = *hit; delete hit; return ret;}
	}
};

const float epsilon = 0.002f;

struct Listener : Intersectable {
	
	vec3 p;
	vec3 n;
	float alpha;
	float h;
	Light* light;
	

	Listener(vec3 p, vec3 dir, float h, float alpha, vec3 Le) {
	
		this->p = p;
		n = dir;
		this->h = h;
		this->alpha = alpha;
		light = new Light(p + n * h/2, Le);
	
	}


	Hit intersect(const Ray& ray) {
		Hit* hit = new Hit();
	
		float a = powf(dot(ray.dir, n),2) - dot(ray.dir, ray.dir) * cosf(alpha) * cosf(alpha);
		float b = 2 * dot(ray.dir, n) * dot(ray.start - p, n) - 2 * (dot(ray.dir, ray.start - p) * powf(cosf(alpha), 2));
		float c = dot(ray.start - p, n) * dot(ray.start - p, n) - dot(ray.start - p, ray.start - p) * powf(cosf(alpha), 2);
	

		float discr = b * b - 4 * a * c;
		if (discr < 0)
			{Hit ret = *hit; delete hit; return ret;}
		float t1 = (-b + sqrtf(discr)) / 2.0f / a;
		float t2 = (-b - sqrtf(discr)) / 2.0f / a;

		
		float t = (((t1) < (t2)) ? (t1) : (t2));

		if (t1 < 0 && t2 < 0) { Hit ret = *hit; delete hit; return ret; }
		else if (t1 < 0 && t2 > 0) t = t2;
		else if (t1 > 0 && t2 < 0) t = t1;
		else t = (((t1) < (t2)) ? (t1) : (t2));
		
		vec3 r;
		
		r = ray.start + ray.dir * t;
		if (dot(r - p, n) <= h && dot(r - p, n) >= 0) {
			hit->t = t;
			hit->position = r;
			hit->normal = (2 * dot((r - p), n) * n - 2 * (r - p) * cosf(alpha) * cosf(alpha));
		}
		else {
			t = t == t1 ? t2 : t1;
			r = ray.start + ray.dir * t;
			if (dot(r - p, n) <= h && dot(r - p, n) >= 0) {
				hit->t = t;
				hit->position = r;
				hit->normal = (2 * dot((r - p), n) * n - 2 * (r - p) * cosf(alpha) * cosf(alpha));
			}
		}

		{ Hit ret = *hit; delete hit; return ret; }
	}

	~Listener() {
		delete light;
	}

};

GPUProgram gpuProgram;

class FullScreenTexturedQuad {
	unsigned int vao;	
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	
		glBindVertexArray(vao);		

		unsigned int vbo;	
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);    
	}

	void Draw() {
		glBindVertexArray(vao);
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Listener*> listeners;
	Camera camera;
	vec3 La;
public:

	

	void build() {
		vec3 eye = 4*vec3(-4* 0.4, 0, -3* 0.4), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.0f, 0.0f, 0.0f);
		

		objects.push_back(new Cube(4, 0, 0, 0));
		objects.push_back(new Icosahedron(0.7, 0, 0, 1.6));
		objects.push_back(new Octahedron(0.7, 0, 0, -1));

		Listener* red = new Listener(vec3(1.300, 0.735, 2), normalize(vec3(0, 0, -1)), 0.3, 20 * M_PI / 180, vec3(1, 0, 0));
		objects.push_back(red);
		listeners.push_back(red);
		
		Listener *blue = new Listener(vec3(-0.455, 0.065, -0.870), normalize(vec3(-0.577, 0.577, 0.577)), 0.4, 20 * M_PI / 180, vec3(0, 1, 0));
		objects.push_back(blue);
		listeners.push_back(blue);

		Listener* green = new Listener(vec3(-0.297, -0.871, 0.5745), normalize(vec3(0, -0.934, -0.357)), 0.3, 10 * M_PI / 180, vec3(0, 0, 1));
		objects.push_back(green);
		listeners.push_back(green);
		
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	
	vec3 trace(Ray ray, int depth = 0) {

		Hit hit = firstIntersect(ray);

		if (hit.t < 0) return La;

		vec3 outRadiance = vec3(1, 1, 1) * (0.2 * (1 - dot(hit.normal, ray.dir)));
		
		for (Listener* listener : listeners) {
			vec3 __hitp = hit.position + hit.normal * epsilon;

			Ray shadowRay(__hitp, listener->light->p - __hitp);
			Hit shadowhit = firstIntersect(shadowRay);
			float l = length(listener->light->p - __hitp);
			float t = shadowhit.t;

			float cosTheta = dot(hit.normal, shadowRay.dir);
			if (cosTheta > 0 && ( l < t || t < 0 )) {
				outRadiance = outRadiance + listener->light->Le / (powf( length(listener->light->p - __hitp) , 2));
			}
		}
		
		return outRadiance;
	}
	
	void relocateListener(int clickX, int clickY) {
		Ray finder = camera.getRay(clickX, windowHeight - clickY);
		Hit hit = firstIntersect(finder);

		float minLen = length(listeners[0]->p - hit.position);
		Listener* closest = listeners[0];
		for (Listener* l : listeners) {
			float temp = length(l->p - hit.position);
			if (temp < minLen) {
				minLen = temp;
				closest = l;
			}
		}

		closest->n = normalize(hit.normal);
		closest->p = hit.position;
		closest->light->p = closest->p + closest->h * closest->n / 2;

		std::vector<vec4> image(windowWidth * windowHeight);
		render(image);
		delete fullScreenTexturedQuad;
		fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
		glutPostRedisplay();
	}
	
};


Scene scene;

const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";


void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();							
}

void onKeyboard(unsigned char key, int pX, int pY) {

}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		scene.relocateListener(pX, pY);
	}
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
}